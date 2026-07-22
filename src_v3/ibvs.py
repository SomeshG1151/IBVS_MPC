import numpy as np

FX, FY = 277.19, 277.19
CX, CY = 320.0, 240.0

GAIN_FAR   = 3.0
GAIN_NEAR  = 2.0
DEPTH_BLEND_THRESHOLD = 2.0

MAX_APPROACH_SPEED = 2.0
MAX_LATERAL_SPEED  = 3.0

DEFAULT_DEPTH  = 3.0
DAMPING        = 0.01


def pixel_to_normalized(corners):
    corners = np.asarray(corners, dtype=float)
    x = (corners[:, 0] - CX) / FX
    y = (corners[:, 1] - CY) / FY
    return np.column_stack((x, y))


def interaction_row(u, v, depth):
    x = (u - CX) / FX
    y = (v - CY) / FY

    row_x = np.array([-1/depth, 0, x/depth, x*y, -(1 + x**2), y])
    row_y = np.array([0, -1/depth, y/depth, 1 + y**2, -x*y, -x])

    return np.vstack((row_x, row_y))


def build_interaction_matrix(corners, depth):
    rows = []
    for u, v in corners:
        rows.append(interaction_row(u, v, depth))
    return np.vstack(rows)


def damped_pseudoinverse(L, damping=DAMPING):
    I = np.eye(L.shape[0])
    return L.T @ np.linalg.inv(L @ L.T + damping * I)


class IBVSController:
    def __init__(self, target_corners):
        self.target_pixels     = np.asarray(target_corners, dtype=float).ravel()
        self.target_normalized = pixel_to_normalized(target_corners).ravel()

    def compute(self, corners, depth=DEFAULT_DEPTH):
        current = pixel_to_normalized(corners).ravel()
        error   = current - self.target_normalized

        depth_ratio = np.clip(depth / DEPTH_BLEND_THRESHOLD, 0.0, 1.0)
        gain        = GAIN_NEAR + (GAIN_FAR - GAIN_NEAR) * depth_ratio

        L        = build_interaction_matrix(corners, depth)
        velocity = -gain * damped_pseudoinverse(L) @ error

        velocity[0] = np.clip(velocity[0] * 1.5, -MAX_LATERAL_SPEED,  MAX_LATERAL_SPEED)
        velocity[1] = np.clip(velocity[1] * 1.5, -MAX_LATERAL_SPEED,  MAX_LATERAL_SPEED)

        alignment   = np.clip(1.0 - np.linalg.norm(error) / 2.5, 0.0, 1.0)
        velocity[2] = np.clip(velocity[2], -MAX_APPROACH_SPEED, MAX_APPROACH_SPEED) * alignment

        return velocity, error, L

    def feature_error_norm(self, corners):
        current = np.asarray(corners, dtype=float).ravel()
        return np.linalg.norm(current - self.target_pixels)


def make_target_corners(width=640, height=480, half_side=80):
    cx, cy = width // 2, height // 2
    return np.array([
        [cx - half_side, cy - half_side],
        [cx + half_side, cy - half_side],
        [cx + half_side, cy + half_side],
        [cx - half_side, cy + half_side],
    ], dtype=float)