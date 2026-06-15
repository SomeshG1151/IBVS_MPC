import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.linalg import block_diag

HORIZON = 10
DT      = 0.1

Q_DIAG = 1.0
R_DIAG = 0.05
P_DIAG = 5.0

VX_MAX =  2.0
VX_MIN = -2.0
VY_MAX =  1.5
VZ_MAX =  1.5

NX = 8
NU = 3


class LinearMPC:

    def __init__(self):
        self.Q = Q_DIAG * np.eye(NX)
        self.R = R_DIAG * np.eye(NU)
        self.P = P_DIAG * np.eye(NX)
        self._last_u = np.zeros(NU)

    def _build_B(self, L):
        return L[:, :3] * DT

    def _build_prediction(self, B):
        Fx = np.tile(np.eye(NX), (HORIZON + 1, 1))
        Fu = np.zeros((NX * (HORIZON + 1), NU * HORIZON))
        for k in range(1, HORIZON + 1):
            row = k * NX
            for j in range(k):
                Fu[row:row+NX, j*NU:j*NU+NU] = B
        return Fx, Fu

    def _build_cost_matrices(self):
        Q_bar = block_diag(*([self.Q] * HORIZON + [self.P]))
        R_bar = block_diag(*([self.R] * HORIZON))
        return Q_bar, R_bar

    def solve(self, e, L):
        B            = self._build_B(L)
        Fx, Fu       = self._build_prediction(B)
        Q_bar, R_bar = self._build_cost_matrices()

        H = Fu.T @ Q_bar @ Fu + R_bar
        H = (H + H.T) / 2
        f = Fu.T @ Q_bar @ Fx @ e

        lb = np.tile([VX_MIN, -VY_MAX, -VZ_MAX], HORIZON)
        ub = np.tile([VX_MAX,  VY_MAX,  VZ_MAX], HORIZON)

        u0 = np.tile(self._last_u, HORIZON)

        result = minimize(
            fun=lambda u: 0.5 * u @ H @ u + f @ u,
            x0=u0,
            jac=lambda u: H @ u + f,
            method="SLSQP",
            bounds=Bounds(lb, ub),
            options={"maxiter": 150, "ftol": 1e-7},
        )

        U_opt = result.x.reshape(HORIZON, NU)
        u_opt = np.clip(U_opt[0], [VX_MIN, -VY_MAX, -VZ_MAX], [VX_MAX, VY_MAX, VZ_MAX])
        self._last_u = u_opt.copy()
        return u_opt

    def predict_trajectory(self, e, L, u_seq):
        B  = self._build_B(L)
        xs = [e.copy()]
        for k in range(HORIZON):
            xs.append(xs[-1] + B @ u_seq[k])
        return np.array(xs)