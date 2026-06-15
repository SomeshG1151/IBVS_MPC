import numpy as np

FX = 277.19
FY = 277.19
CX = 320.0
CY = 240.0

LAMBDA     = 0.8
DEPTH_INIT = 3.0


def interaction_row(u: float, v: float, Z: float) -> np.ndarray:
    x = u - CX
    y = v - CY

    Lu = np.array([
        -FX / Z,
         0,
         x / Z,
         x * y / FY,
        -(FX**2 + x**2) / FX,
         y * FX / FY,
    ])
    Lv = np.array([
         0,
        -FY / Z,
         y / Z,
        -(FY**2 + y**2) / FY,
         x * y / FX,
        -x * FY / FX,
    ])
    return np.vstack([Lu, Lv])


def build_interaction_matrix(corners: np.ndarray, Z: float) -> np.ndarray:
    return np.vstack([interaction_row(u, v, Z) for u, v in corners])


class IBVSController:

    def __init__(self, target_corners: np.ndarray):
        self.s_star = target_corners.astype(float).ravel()

    def compute(self, corners: np.ndarray, Z: float = DEPTH_INIT):
        s  = corners.astype(float).ravel()
        e  = s - self.s_star
        L  = build_interaction_matrix(corners, Z)
        Lp = np.linalg.pinv(L)
        vc = -LAMBDA * Lp @ e
        return vc, e, L

    def feature_error_norm(self, corners: np.ndarray) -> float:
        return float(np.linalg.norm(corners.ravel() - self.s_star))

    @property
    def target(self) -> np.ndarray:
        return self.s_star.reshape(4, 2)


def make_target_corners(W: int = 640, H: int = 480, half_side_px: int = 80) -> np.ndarray:
    cx, cy = W // 2, H // 2
    d = half_side_px
    return np.array([
        [cx - d, cy - d],
        [cx + d, cy - d],
        [cx + d, cy + d],
        [cx - d, cy + d],
    ], dtype=float)