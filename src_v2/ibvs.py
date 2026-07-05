import numpy as np

FX, FY = 277.19, 277.19
CX, CY = 320.0, 240.0

LAMBDA = 0.3
DEPTH_INIT = 3.0
DAMPING_MU = 0.01


def to_normalized(corners):
    corners = np.asarray(corners, dtype=float)
    x = (corners[:, 0] - CX) / FX
    y = (corners[:, 1] - CY) / FY
    return np.column_stack((x, y))


def interaction_row(u, v, Z):
    x = (u - CX) / FX
    y = (v - CY) / FY

    Lu = np.array([
        -1 / Z,
        0,
        x / Z,
        x * y,
        -(1 + x**2),
        y
    ])

    Lv = np.array([
        0,
        -1 / Z,
        y / Z,
        1 + y**2,
        -x * y,
        -x
    ])

    return np.vstack((Lu, Lv))


def build_interaction_matrix(corners, Z):
    rows = []
    for u, v in corners:
        rows.append(interaction_row(u, v, Z))
    return np.vstack(rows)


def damped_pinv(L, mu=DAMPING_MU):
    I = np.eye(L.shape[0])
    return L.T @ np.linalg.inv(L @ L.T + mu * I)


class IBVSController:
    def __init__(self, target_corners):
        self.target_corners_px = np.asarray(
            target_corners, dtype=float
        ).ravel()

        self.s_star = to_normalized(
            target_corners
        ).ravel()

    def compute(self, corners, Z=DEPTH_INIT):
        s = to_normalized(corners).ravel()
        error = s - self.s_star

        L = build_interaction_matrix(corners, Z)
        velocity = -LAMBDA * damped_pinv(L) @ error

        return velocity, error, L

    def feature_error_norm(self, corners):
        corners = np.asarray(corners, dtype=float).ravel()
        return np.linalg.norm(
            corners - self.target_corners_px
        )


def make_target_corners(width=640, height=480, half_side=80):
    cx = width // 2
    cy = height // 2
    d = half_side

    return np.array([
        [cx - d, cy - d],
        [cx + d, cy - d],
        [cx + d, cy + d],
        [cx - d, cy + d]
    ], dtype=float)