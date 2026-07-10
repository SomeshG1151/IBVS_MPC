import sys, time, threading
import cv2
import numpy as np
from pymavlink import mavutil
import gz.transport13 as transport
from gz.msgs10 import image_pb2

from tracker import Tracker
from ibvs import IBVSController, make_target_corners
from mpc import LinearMPC, NX, NU_Z_HOV

CAMERA_TOPIC = "/world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image"
MAVLINK_URI = "udpin:localhost:14550"

FRAME_W = 640
FRAME_H = 480

TAKEOFF_ALT = 5.0
REAL_TARGET_WIDTH = 1.0

LOST_LIMIT = 30
IMPACT_RATIO = 0.60

MPC_RATE = 0.15
VC_CLIP = 0.9

TARGET_CORNERS = make_target_corners(FRAME_W, FRAME_H, half_side=80)

latest_frame = [None]
frame_counter = [0]
frame_lock = threading.Lock()

_z_filt = [5.0]


def on_camera_message(msg):
    try:
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        with frame_lock:
            latest_frame[0] = frame
            frame_counter[0] += 1

    except Exception as e:
        print(f"[CAM] {e}")


def get_frame():
    with frame_lock:
        return latest_frame[0]


def get_frame_and_id():
    with frame_lock:
        return latest_frame[0], frame_counter[0]


def start_camera():
    node = transport.Node()
    ok = node.subscribe(image_pb2.Image, CAMERA_TOPIC, on_camera_message)

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


def arm_and_takeoff(conn, altitude):
    conn.set_mode("GUIDED")
    time.sleep(1)

    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0,
    )

    conn.motors_armed_wait()
    print("[MAV] Armed!")

    time.sleep(2)

    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude,
    )

    print(f"[MAV] Taking off to {altitude} m ...")

    while True:
        msg = conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)

        if msg and msg.relative_alt / 1000.0 >= altitude - 0.5:
            print(f"[MAV] Reached {msg.relative_alt / 1000:.2f} m")
            break

        time.sleep(0.5)

    conn.set_mode("GUIDED_NOGPS")
    time.sleep(0.5)
    print("[MAV] GUIDED_NOGPS — MPC control active")


_state = np.zeros(NX)
_alt = [TAKEOFF_ALT]

_state_lock = threading.Lock()
_alt_lock = threading.Lock()

_vel_debug_count = [0]


def _state_reader(conn, stop_event):
    while not stop_event.is_set():
        msg = conn.recv_match(type=["GLOBAL_POSITION_INT", "ATTITUDE"], blocking=True, timeout=0.05)

        if msg is None:
            continue

        with _state_lock:
            if msg.get_type() == "GLOBAL_POSITION_INT":
                _state[0] = msg.vx / 100.0
                _state[1] = msg.vy / 100.0
                _state[2] = msg.vz / 100.0

                if _vel_debug_count[0] < 100:
                    print(f"[RAW VEL] vx={msg.vx} vy={msg.vy} vz={msg.vz}")
                    _vel_debug_count[0] += 1

                with _alt_lock:
                    _alt[0] = max(msg.relative_alt / 1000.0, 0.5)

            elif msg.get_type() == "ATTITUDE":
                _state[3] = msg.roll
                _state[4] = msg.pitch
                _state[5] = msg.yaw


def get_state():
    with _state_lock:
        return _state.copy()


def get_altitude():
    with _alt_lock:
        return _alt[0]


_att_cmd = [0.0, 0.0, 0.65]
_att_lock = threading.Lock()


def euler_to_quat(roll, pitch, yaw=0.0):
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)

    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def send_attitude(conn, roll, pitch, thrust):
    roll = np.clip(roll, -0.55, 0.55)
    pitch = np.clip(pitch, -0.50, 0.50)
    thrust = np.clip(thrust, 0.30, 0.85)

    conn.mav.set_attitude_target_send(
        0,
        conn.target_system,
        conn.target_component,
        0b00000111,
        euler_to_quat(roll, pitch),
        0, 0, 0,
        float(thrust),
    )


def att_heartbeat(conn, stop_event):
    while not stop_event.is_set():
        with _att_lock:
            roll, pitch, thrust = _att_cmd

        send_attitude(conn, roll, pitch, thrust)
        time.sleep(0.05)


def set_attitude(conn, roll, pitch, thrust):
    with _att_lock:
        _att_cmd[0] = roll
        _att_cmd[1] = pitch
        _att_cmd[2] = thrust

    send_attitude(conn, roll, pitch, thrust)


_fc = [0]
_lp = [time.time()]


def dbg(**kw):
    _fc[0] += 1

    if time.time() - _lp[0] >= 0.5:
        _lp[0] = time.time()
        text = "  ".join(f"{k}={v}" for k, v in kw.items())
        print(f"[{_fc[0]:05d}] {text}")


def main():
    conn = connect_mavlink(MAVLINK_URI)
    arm_and_takeoff(conn, TAKEOFF_ALT)

    stop_event = threading.Event()
    threading.Thread(target=_state_reader, args=(conn, stop_event), daemon=True).start()
    threading.Thread(target=att_heartbeat, args=(conn, stop_event), daemon=True).start()

    node = start_camera()
    timeout = time.time() + 10
    while get_frame() is None:
        if time.time() > timeout:
            print("[CAM] No frame after 10s")
            stop_event.set()
            sys.exit(1)
        time.sleep(0.1)
    print("[CAM] OK — starting MPC-IBVS loop")

    tracker = Tracker()
    ibvs = IBVSController(target_corners=TARGET_CORNERS)
    mpc = LinearMPC()
    acquired = False

    last_seen_frame_id = -1
    last_solve_t = 0.0
    u_opt_held = np.array([0.0, 0.0, 0.0, NU_Z_HOV])

    try:
        while True:
            frame, frame_id = get_frame_and_id()
            if frame is None:
                continue

            if frame_id == last_seen_frame_id:
                time.sleep(0.005)
                continue
            last_seen_frame_id = frame_id

            frame = frame.copy()

            corners, centroid, bbox, valid = tracker.process(frame)
            x_now = get_state()

            if not valid or tracker.lost > LOST_LIMIT:
                set_attitude(conn, 0.0, 0.0, 0.65)
                label = "WAITING..." if not acquired else "LOST — HOVERING"
                color = (0, 165, 255) if not acquired else (0, 0, 255)
                cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                dbg(status="LOST" if acquired else "WAITING", lost=tracker.lost)

            else:
                acquired = True
                x, y, w, h = bbox
                area_ratio = (w * h) / (FRAME_W * FRAME_H)

                if area_ratio >= IMPACT_RATIO:
                    print("[CTRL] IMPACT")
                    set_attitude(conn, 0.0, 0.0, 0.30)
                    break

                Z_est_raw = np.clip((REAL_TARGET_WIDTH * 277.19) / max(w, 1), 0.5, 20.0)
                _z_filt[0] = 0.3 * Z_est_raw + 0.7 * _z_filt[0]
                Z_est = _z_filt[0]

                vc, e, L = ibvs.compute(corners, Z=Z_est)
                vc_clipped = np.clip(vc, -VC_CLIP, VC_CLIP)
                x_star = mpc.compute_reference(vc_clipped, roll=x_now[3], pitch=x_now[4], yaw=x_now[5])

                print(f"[VC_RAW] {vc[:3].round(4)}  x_star={x_star[:3].round(4)}")

                now = time.time()
                if now - last_solve_t >= MPC_RATE:
                    u_opt_held = mpc.solve(x_now, x_star)
                    last_solve_t = now

                u_opt = u_opt_held
                roll, pitch, thrust = mpc.to_attitude_command(u_opt)
                set_attitude(conn, roll, pitch, thrust)

                err = ibvs.feature_error_norm(corners)
                roll_cmd, pitch_cmd, yaw_rate_cmd, nu_z = u_opt

                dbg(
                    err=f"{err:.1f}px",
                    area=f"{area_ratio:.3f}",
                    yaw=f"{np.degrees(x_now[5]):.1f}°",
                    roll_cmd=f"{np.degrees(roll_cmd):.1f}°",
                    pitch_cmd=f"{np.degrees(pitch_cmd):.1f}°",
                    roll_actual=f"{np.degrees(x_now[3]):.1f}°",
                    pitch_actual=f"{np.degrees(x_now[4]):.1f}°",
                    nuz=f"{nu_z:.2f}N",
                    roll=f"{np.degrees(roll):.1f}°",
                    pitch=f"{np.degrees(pitch):.1f}°",
                    thrust=f"{thrust:.3f}",
                )

                frame = Tracker.draw(frame, corners, centroid, bbox, area_ratio)
                cv2.putText(frame, f"err={err:.1f}px  area={area_ratio:.3f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.putText(frame, f"r={np.degrees(roll):.1f}  p={np.degrees(pitch):.1f}  t={thrust:.3f}",
                            (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 0), 2)
                cv2.putText(frame, f"cmd=[{np.degrees(roll_cmd):.1f},{np.degrees(pitch_cmd):.1f}]  nuz={nu_z:.1f}N",
                            (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 2)
                for pt in TARGET_CORNERS.astype(int):
                    cv2.drawMarker(frame, tuple(pt), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
                cv2.drawMarker(frame, (FRAME_W // 2, FRAME_H // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

            cv2.imshow("MPC-IBVS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[CTRL] Stopped.")
    finally:
        stop_event.set()
        set_attitude(conn, 0.0, 0.0, 0.30)
        time.sleep(0.5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()