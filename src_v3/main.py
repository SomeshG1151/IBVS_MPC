import sys, time, threading, collections
import cv2
import numpy as np
from pymavlink import mavutil
import gz.transport13 as transport
from gz.msgs10 import image_pb2

from tracker import Tracker
from ibvs import IBVSController, make_target_corners
from mpc import LinearMPC, NX, NU_Z_HOV

CAMERA_TOPIC  = "/world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image"
MAVLINK_URI   = "udpin:localhost:14550"

FRAME_W = 640
FRAME_H = 480

TAKEOFF_ALT       = 5.0
TARGET_REAL_WIDTH = 1.0

LOST_FRAME_LIMIT  = 30
IMPACT_AREA_RATIO = 0.60

MPC_SOLVE_PERIOD  = 0.02
VELOCITY_CLIP     = 10

DEPTH_FILTER_ALPHA = 0.3
FOCAL_LENGTH_X     = 277.19

TARGET_CORNERS = make_target_corners(FRAME_W, FRAME_H, half_side=80)
TARGET_CORNERS[:, 1] += 100

latest_frame  = [None]
frame_counter = [0]
frame_lock    = threading.Lock()

depth_filtered = [5.0]

_vehicle_state = np.zeros(NX)
_altitude      = [TAKEOFF_ALT]
_state_lock    = threading.Lock()
_alt_lock      = threading.Lock()

_attitude_cmd  = [0.0, 0.0, 0.65]
_att_cmd_lock  = threading.Lock()

_debug_frame_count  = [0]
_debug_last_print   = [time.time()]


def on_camera_frame(msg):
    try:
        img   = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with frame_lock:
            latest_frame[0]   = frame
            frame_counter[0] += 1
    except Exception as e:
        print(f"[CAM] {e}")


def get_latest_frame():
    with frame_lock:
        return latest_frame[0]


def get_frame_and_id():
    with frame_lock:
        return latest_frame[0], frame_counter[0]


def start_camera_subscriber():
    node = transport.Node()
    ok   = node.subscribe(image_pb2.Image, CAMERA_TOPIC, on_camera_frame)
    if not ok:
        print("[CAM] Subscribe failed")
        sys.exit(1)
    print("[CAM] Subscribed")
    return node


def connect_mavlink(uri):
    conn = mavutil.mavlink_connection(uri)
    conn.wait_heartbeat()
    print(f"[MAV] Connected to system {conn.target_system}")
    return conn


def request_message_interval(conn, message_id, hz):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0, message_id, int(1e6 / hz), 0, 0, 0, 0, 0,
    )


def arm_and_takeoff(conn, altitude):
    conn.set_mode("GUIDED")
    time.sleep(1)

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0,
    )
    conn.motors_armed_wait()
    print("[MAV] Armed")
    time.sleep(2)

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude,
    )
    print(f"[MAV] Taking off to {altitude} m")

    while True:
        msg = conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        if msg and msg.relative_alt / 1000.0 >= altitude - 0.5:
            print(f"[MAV] Reached {msg.relative_alt / 1000:.2f} m")
            break
        time.sleep(0.5)

    conn.set_mode("GUIDED_NOGPS")
    time.sleep(0.5)
    print("[MAV] GUIDED_NOGPS active")


def euler_to_quaternion(roll, pitch, yaw=0.0):
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    return [
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]


def send_attitude_setpoint(conn, roll, pitch, thrust):
    roll   = np.clip(roll,   -0.7,  0.7)
    pitch  = np.clip(pitch,  -0.65, 0.65)
    conn.mav.set_attitude_target_send(
        0, conn.target_system, conn.target_component,
        0b00000111,
        euler_to_quaternion(roll, pitch),
        0, 0, 0, float(thrust),
    )


def set_attitude(conn, roll, pitch, thrust):
    with _att_cmd_lock:
        _attitude_cmd[0] = roll
        _attitude_cmd[1] = pitch
        _attitude_cmd[2] = thrust
    send_attitude_setpoint(conn, roll, pitch, thrust)


def attitude_heartbeat_thread(conn, stop_event):
    while not stop_event.is_set():
        with _att_cmd_lock:
            roll, pitch, thrust = _attitude_cmd
        send_attitude_setpoint(conn, roll, pitch, thrust)
        time.sleep(0.05)


def state_reader_thread(conn, stop_event):
    while not stop_event.is_set():
        msg = conn.recv_match(
            type=["GLOBAL_POSITION_INT", "ATTITUDE"],
            blocking=True, timeout=0.05,
        )
        if msg is None:
            continue

        with _state_lock:
            if msg.get_type() == "GLOBAL_POSITION_INT":
                _vehicle_state[0] = msg.vx / 100.0
                _vehicle_state[1] = msg.vy / 100.0
                _vehicle_state[2] = msg.vz / 100.0
                with _alt_lock:
                    _altitude[0] = max(msg.relative_alt / 1000.0, 0.5)
            elif msg.get_type() == "ATTITUDE":
                _vehicle_state[3] = msg.roll
                _vehicle_state[4] = msg.pitch
                _vehicle_state[5] = msg.yaw


def get_vehicle_state():
    with _state_lock:
        return _vehicle_state.copy()


def get_altitude():
    with _alt_lock:
        return _altitude[0]


def log(**kwargs):
    _debug_frame_count[0] += 1
    if time.time() - _debug_last_print[0] >= 0.5:
        _debug_last_print[0] = time.time()
        print(f"[{_debug_frame_count[0]:05d}] " + "  ".join(f"{k}={v}" for k, v in kwargs.items()))


def main():
    conn = connect_mavlink(MAVLINK_URI)
    request_message_interval(conn, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 50)
    request_message_interval(conn, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50)
    arm_and_takeoff(conn, TAKEOFF_ALT)

    stop_event = threading.Event()
    threading.Thread(target=state_reader_thread,     args=(conn, stop_event), daemon=True).start()
    threading.Thread(target=attitude_heartbeat_thread, args=(conn, stop_event), daemon=True).start()

    camera_node = start_camera_subscriber()
    deadline    = time.time() + 10
    while get_latest_frame() is None:
        if time.time() > deadline:
            print("[CAM] No frame after 10s")
            stop_event.set()
            sys.exit(1)
        time.sleep(0.1)
    print("[CAM] First frame received — starting control loop")

    tracker  = Tracker()
    ibvs     = IBVSController(target_corners=TARGET_CORNERS)
    mpc      = LinearMPC()
    acquired = False

    last_frame_id  = -1
    last_solve_t   = 0.0
    held_input     = np.array([0.0, 0.0, 0.0, NU_Z_HOV])

    try:
        while True:
            frame, frame_id = get_frame_and_id()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.005)
                continue
            last_frame_id = frame_id
            frame = frame.copy()

            corners, centroid, bbox, target_visible = tracker.process(frame)
            vehicle_state = get_vehicle_state()

            if not target_visible or tracker.lost > LOST_FRAME_LIMIT:
                set_attitude(conn, 0.0, 0.0, 0.65)
                label = "WAITING..." if not acquired else "LOST — HOVERING"
                color = (0, 165, 255)  if not acquired else (0, 0, 255)
                cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                log(status="WAITING" if not acquired else "LOST", lost=tracker.lost)

            else:
                acquired = True
                x, y, w, h  = bbox
                area_ratio  = (w * h) / (FRAME_W * FRAME_H)

                if area_ratio >= IMPACT_AREA_RATIO:
                    print("[CTRL] IMPACT — cutting thrust")
                    set_attitude(conn, 0.0, 0.0, 0.30)
                    break

                depth_raw      = np.clip((TARGET_REAL_WIDTH * FOCAL_LENGTH_X) / max(w, 1), 0.5, 20.0)
                depth_filtered[0] = DEPTH_FILTER_ALPHA * depth_raw + (1 - DEPTH_FILTER_ALPHA) * depth_filtered[0]
                depth          = depth_filtered[0]

                camera_vel, error, L = ibvs.compute(corners, depth)
                camera_vel_clipped   = np.clip(camera_vel, -VELOCITY_CLIP, VELOCITY_CLIP)

                roll, pitch, yaw     = vehicle_state[3], vehicle_state[4], vehicle_state[5]
                reference_state      = mpc.compute_reference(camera_vel_clipped, roll, pitch, yaw)

                now = time.time()
                if now - last_solve_t >= MPC_SOLVE_PERIOD:
                    held_input   = mpc.solve(vehicle_state, reference_state)
                    last_solve_t = now

                roll_out, pitch_out, thrust_out = mpc.to_attitude_command(held_input)
                set_attitude(conn, roll_out, pitch_out, thrust_out)

                roll_cmd, pitch_cmd, _, thrust_force = held_input
                log(
                    err=f"{ibvs.feature_error_norm(corners):.1f}px",
                    area=f"{area_ratio:.3f}",
                    depth=f"{depth:.2f}m",
                    roll_cmd=f"{np.degrees(roll_cmd):.1f}°",
                    pitch_cmd=f"{np.degrees(pitch_cmd):.1f}°",
                    thrust=f"{thrust_out:.3f}",
                )

                frame = Tracker.draw(frame, corners, centroid, bbox, area_ratio)
                cv2.putText(frame, f"err={ibvs.feature_error_norm(corners):.1f}px  area={area_ratio:.3f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.putText(frame, f"r={np.degrees(roll_out):.1f}  p={np.degrees(pitch_out):.1f}  t={thrust_out:.3f}",
                            (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 0), 2)
                for pt in TARGET_CORNERS.astype(int):
                    cv2.drawMarker(frame, tuple(pt), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
                cv2.drawMarker(frame, (FRAME_W//2, FRAME_H//2), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

            cv2.imshow("MPC-IBVS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[CTRL] Stopped")
    finally:
        stop_event.set()
        set_attitude(conn, 0.0, 0.0, 0.30)
        time.sleep(0.5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()