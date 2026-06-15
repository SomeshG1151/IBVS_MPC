import sys
import time
import threading

import cv2
import numpy as np
from pymavlink import mavutil

import gz.transport13 as transport
from gz.msgs10 import image_pb2

from tracker import Tracker
from ibvs import IBVSController, make_target_corners
from mpc import LinearMPC


CAMERA_TOPIC = "/world/iris_runway/model/iris_with_gimbal/model/gimbal/link/pitch_link/sensor/camera/image"
MAVLINK_URI  = "udpin:localhost:14550"

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TAKEOFF_ALT  = 5.0
LOST_LIMIT   = 30
IMPACT_RATIO = 0.60

MAX_ROLL     =  0.40
MAX_PITCH    =  0.35
HOVER_THRUST =  0.65
ROLL_SCALE   =  0.50
ROLL_KP_CX   =  0.0008


def get_forward_pitch(area_ratio: float) -> float:
    if area_ratio < 0.05:
        return -0.28
    elif area_ratio < 0.15:
        return -0.20
    elif area_ratio < 0.30:
        return -0.12
    else:
        return -0.06


TARGET_CORNERS = make_target_corners(FRAME_WIDTH, FRAME_HEIGHT, half_side_px=80)

latest_frame = [None]
frame_lock   = threading.Lock()


def on_camera_message(msg):
    try:
        img   = np.frombuffer(msg.data, dtype=np.uint8)
        img   = img.reshape((msg.height, msg.width, 3))
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with frame_lock:
            latest_frame[0] = frame
    except Exception as e:
        print(f"[CAM] Frame error: {e}")


def get_frame():
    with frame_lock:
        return latest_frame[0]


def start_camera():
    node = transport.Node()
    ok   = node.subscribe(image_pb2.Image, CAMERA_TOPIC, on_camera_message)
    if not ok:
        print(f"[CAM] Failed to subscribe to {CAMERA_TOPIC}")
        sys.exit(1)
    print("[CAM] Subscribed to camera topic")
    return node


def connect_mavlink(uri: str):
    print(f"[MAV] Connecting to {uri} ...")
    conn = mavutil.mavlink_connection(uri)
    conn.wait_heartbeat()
    print(f"[MAV] Heartbeat from system {conn.target_system}")
    return conn


def arm_and_takeoff(conn, altitude: float):
    conn.set_mode("GUIDED")
    time.sleep(1)

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0,
    )
    conn.motors_armed_wait()
    print("[MAV] Armed!")
    time.sleep(2)

    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude,
    )
    print(f"[MAV] Taking off to {altitude} m ...")

    while True:
        msg = conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        if msg:
            rel_alt = msg.relative_alt / 1000.0
            print(f"[MAV] Altitude: {rel_alt:.2f} m")
            if rel_alt >= (altitude - 0.5):
                print(f"[MAV] Reached {rel_alt:.2f} m")
                break
        time.sleep(0.5)

    conn.set_mode("GUIDED_NOGPS")
    time.sleep(0.5)
    print("[MAV] Switched to GUIDED_NOGPS")


_alt_lock = threading.Lock()
_alt_val  = [5.0]


def _alt_reader(conn, stop_event):
    while not stop_event.is_set():
        msg = conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if msg:
            with _alt_lock:
                _alt_val[0] = max(msg.relative_alt / 1000.0, 0.5)


def get_altitude():
    with _alt_lock:
        return _alt_val[0]


def euler_to_quat(roll, pitch, yaw=0.0):
    cy = np.cos(yaw / 2);  sy = np.sin(yaw / 2)
    cr = np.cos(roll / 2); sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2); sp = np.sin(pitch / 2)
    return [
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]


def send_attitude(conn, roll, pitch, thrust):
    roll   = float(np.clip(roll,   -MAX_ROLL,  MAX_ROLL))
    pitch  = float(np.clip(pitch,  -MAX_PITCH, MAX_PITCH))
    thrust = float(np.clip(thrust,  0.30, 0.85))
    q      = euler_to_quat(roll, pitch)
    conn.mav.set_attitude_target_send(
        0,
        conn.target_system,
        conn.target_component,
        0b00000100,
        q,
        0, 0, 0,
        thrust,
    )


att_cmd  = [0.0, 0.0, HOVER_THRUST]
att_lock = threading.Lock()


def att_heartbeat(conn, stop_event):
    while not stop_event.is_set():
        with att_lock:
            r, p, t = att_cmd
        send_attitude(conn, r, p, t)
        time.sleep(0.05)


def set_attitude(conn, roll, pitch, thrust):
    with att_lock:
        att_cmd[0] = roll
        att_cmd[1] = pitch
        att_cmd[2] = thrust
    send_attitude(conn, roll, pitch, thrust)


frame_count = [0]
last_print  = [time.time()]
PRINT_EVERY = 0.5


def debug_print(**kwargs):
    now = time.time()
    frame_count[0] += 1
    if now - last_print[0] >= PRINT_EVERY:
        last_print[0] = now
        parts = "  ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"[{frame_count[0]:05d}] {parts}")


def main():
    conn = connect_mavlink(MAVLINK_URI)
    arm_and_takeoff(conn, TAKEOFF_ALT)

    stop_event = threading.Event()

    threading.Thread(target=_alt_reader, args=(conn, stop_event), daemon=True).start()
    threading.Thread(target=att_heartbeat, args=(conn, stop_event), daemon=True).start()
    print("[ATT] Heartbeat started at 20 Hz")

    node = start_camera()
    print("[CAM] Waiting for first frame ...")
    timeout = time.time() + 10
    while get_frame() is None:
        if time.time() > timeout:
            print("[CAM] No frame after 10s — did you run enable_streaming?")
            stop_event.set()
            sys.exit(1)
        time.sleep(0.1)
    print("[CAM] Camera OK")

    tracker  = Tracker()
    ibvs     = IBVSController(target_corners=TARGET_CORNERS)
    mpc      = LinearMPC()
    acquired = False

    print("[CTRL] MPC+IBVS intercept running (press Q to quit)")

    try:
        while True:
            frame = get_frame()
            if frame is None:
                continue
            frame = frame.copy()

            corners, centroid, bbox, valid = tracker.process(frame)
            Z = get_altitude()

            if not valid or tracker.lost > LOST_LIMIT:
                set_attitude(conn, 0.0, 0.0, HOVER_THRUST)
                status = "WAITING" if not acquired else "LOST"
                label  = "WAITING FOR TARGET..." if not acquired else "TARGET LOST - HOVERING"
                color  = (0, 165, 255) if not acquired else (0, 0, 255)
                cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                debug_print(status=status, lost=tracker.lost, Z=f"{Z:.2f}m")

            else:
                acquired   = True
                x, y, w, h = bbox
                area_ratio = (w * h) / (FRAME_WIDTH * FRAME_HEIGHT)
                cx, cy     = centroid

                if area_ratio >= IMPACT_RATIO:
                    print("[CTRL] *** IMPACT — sphere hit! ***")
                    set_attitude(conn, 0.0, 0.0, 0.30)
                    break

                _, e, L    = ibvs.compute(corners, Z=Z)
                u_opt      = mpc.solve(e, L)
                vx, vy, vz = u_opt

                error_x = cx - FRAME_WIDTH / 2
                roll    = float(np.clip(vy * ROLL_SCALE + error_x * ROLL_KP_CX, -MAX_ROLL, MAX_ROLL))
                pitch   = get_forward_pitch(area_ratio)

                error_y = (cy - FRAME_HEIGHT / 2) / (FRAME_HEIGHT / 2)
                thrust  = float(np.clip(HOVER_THRUST - error_y * 0.10, 0.30, 0.85))

                set_attitude(conn, roll, pitch, thrust)

                err_norm = ibvs.feature_error_norm(corners)

                debug_print(
                    status="INTERCEPT",
                    cx=f"{cx:.1f}", cy=f"{cy:.1f}",
                    ex=f"{error_x:.1f}px",
                    area=f"{area_ratio:.3f}",
                    err=f"{err_norm:.1f}px",
                    Z=f"{Z:.2f}m",
                    roll=f"{np.degrees(roll):.2f}deg",
                    pitch=f"{np.degrees(pitch):.2f}deg",
                    thrust=f"{thrust:.3f}",
                    vy=f"{vy:.3f}", vz=f"{vz:.3f}",
                )

                frame = Tracker.draw(frame, corners, centroid, bbox, area_ratio)
                cv2.putText(frame, f"err={err_norm:.1f}px  area={area_ratio:.3f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.putText(frame, f"r={np.degrees(roll):.1f}  p={np.degrees(pitch):.1f}  t={thrust:.2f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 0), 2)
                cv2.putText(frame, f"vy={vy:.3f}  ex={error_x:.0f}px  Z={Z:.1f}m",
                            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 0), 2)

                for pt in TARGET_CORNERS.astype(int):
                    cv2.drawMarker(frame, tuple(pt), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
                cv2.drawMarker(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
                               (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

            cv2.imshow("MPC-IBVS INTERCEPT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[CTRL] Interrupted.")

    finally:
        print("[CTRL] Stopping.")
        stop_event.set()
        set_attitude(conn, 0.0, 0.0, 0.30)
        time.sleep(0.5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()