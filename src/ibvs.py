import numpy as np

FX, FY = 277.19, 277.19
CX, CY = 320.0, 240.0
LAMBDA     = 0.3
DEPTH_INIT = 3.0

def interaction_row(u, v, Z):
    x = (u - CX) / FX
    y = (v - CY) / FY
    Lu = np.array([-1/Z, 0, x/Z, x*y, -(1+x**2), y])
    Lv = np.array([0, -1/Z, y/Z, 1+y**2, -x*y, -x])
    return np.vstack([Lu, Lv])

def build_interaction_matrix(corners, Z):
    return np.vstack([interaction_row(u, v, Z) for u, v in corners])

class IBVSController:
    def __init__(self, target_corners):
        self.s_star = target_corners.astype(float).ravel()

    def compute(self, corners, Z=DEPTH_INIT):
        s  = corners.astype(float).ravel()
        e  = s - self.s_star
        L  = build_interaction_matrix(corners, Z)
        vc = -LAMBDA * np.linalg.pinv(L) @ e   # eq. 4
        return vc, e, L

    def feature_error_norm(self, corners):
        return float(np.linalg.norm(corners.ravel() - self.s_star))

def make_target_corners(W=640, H=480, half_side_px=80):
    cx, cy, d = W//2, H//2, half_side_px
    return np.array([[cx-d,cy-d],[cx+d,cy-d],[cx+d,cy+d],[cx-d,cy+d]], dtype=float)