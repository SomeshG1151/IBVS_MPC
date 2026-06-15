import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize, Bounds

MASS = 1.73
MX, MY, MZ = 0.04, 0.04, 0.10
AL = 0.17
G = 9.81
H = 0.01
N = 10
NX, NU = 9, 4

def _build_matrices():
    Ac = np.zeros((9, 9))
    Ac[0:3, 3:6] = np.array([[0,-G,0],[G,0,0],[0,0,0]])
    Ac[3:6, 6:9] = np.eye(3)
    A = np.eye(9) + H * Ac
    h_roll  =  AL / (4 * MX)
    h_pitch = -AL / (4 * MY)
    B = np.zeros((9, 4))
    B[0, 1] = H * h_pitch
    B[1, 0] = H * h_roll
    B[2, 3] = H / MASS
    B[3:6, 0:3] = 0.5 * H * np.diag([1/MX, 1/MY, 1/MZ])
    B[6:9, 0:3] = H * np.diag([1/MX, 1/MY, 1/MZ])
    return A, B

NU_Z_HOV = MASS * G
TAU_MAX  = (NU_Z_HOV / 4) * AL
U_LB = np.array([-TAU_MAX, -TAU_MAX, -TAU_MAX, 0.0])
U_UB = np.array([ TAU_MAX,  TAU_MAX,  TAU_MAX, 4*G*MASS])
ANGLE_LIM = np.pi / 9
X_LB = np.full(9, -np.inf); X_LB[3:6] = -ANGLE_LIM
X_UB = np.full(9,  np.inf); X_UB[3:6] =  ANGLE_LIM

class LinearMPC:
    def __init__(self):
        self.A, self.B = _build_matrices()
        self.Qx = np.diag([
            2.0, 2.0, 2.0,    # vx, vy, vz
            20.0, 20.0, 0.1,  # θ, φ, ψ
            0.5, 0.5, 0.1     # ωx, ωy, ωz
        ])
        self.Qu = np.diag([0.5, 0.5, 0.1, 2.0])
        try:
            self.Qf = solve_discrete_are(self.A, self.B, self.Qx, self.Qu)
        except Exception:
            print("[MPC] Riccati failed, using 10*Qx")
            self.Qf = 10 * self.Qx
        self._last_u = np.array([0.0, 0.0, 0.0, NU_Z_HOV])
        print(f"[MPC] TAU_MAX={TAU_MAX:.3f} N·m  NU_Z_HOV={NU_Z_HOV:.2f} N")
        print(f"[MPC] A norm={np.linalg.norm(self.A):.3f}  B norm={np.linalg.norm(self.B):.3f}")

    def compute_reference(self, vc: np.ndarray) -> np.ndarray:
        # Forward-facing camera: cam-z=body-x, cam-x=-body-y, cam-y=-body-z
        R_Bc = np.array([[ 0, 0, 1],
                         [-1, 0, 0],
                         [ 0,-1, 0]], dtype=float)
        v_body = np.clip(R_Bc @ vc[:3], -1.5, 1.5)
        x_star = np.zeros(NX)
        x_star[0:3] = v_body
        # attitude ref = 0 (hover), angular rate ref = 0
        return x_star

    def _build_condensed_qp(self):
        from scipy.linalg import block_diag
        A, B = self.A, self.B
        Fx = np.zeros(((N+1)*NX, NX))
        Fu = np.zeros(((N+1)*NX, N*NU))
        Ak = np.eye(NX)
        for k in range(N+1):
            Fx[k*NX:(k+1)*NX] = Ak
            Ak = A @ Ak
        for k in range(1, N+1):
            for j in range(k):
                Fu[k*NX:(k+1)*NX, j*NU:(j+1)*NU] = np.linalg.matrix_power(A, k-1-j) @ B
        Q_bar = block_diag(*([self.Qx]*N + [self.Qf]))
        R_bar = block_diag(*([self.Qu]*N))
        H_mat = Fu.T @ Q_bar @ Fu + R_bar
        H_mat = (H_mat + H_mat.T) / 2
        return H_mat, Fu, Fx, Q_bar

    def solve(self, x_current: np.ndarray, x_star: np.ndarray) -> np.ndarray:
        x_tilde = x_current - x_star
        H_mat, Fu, Fx, Q_bar = self._build_condensed_qp()
        f = Fu.T @ Q_bar @ Fx @ x_tilde
        u0 = np.tile(self._last_u, N)
        result = minimize(
            fun=lambda u: 0.5 * u @ H_mat @ u + f @ u,
            x0=u0,
            jac=lambda u: H_mat @ u + f,
            method="SLSQP",
            bounds=Bounds(np.tile(U_LB, N), np.tile(U_UB, N)),
            options={"maxiter": 300, "ftol": 1e-7},
        )
        if not result.success:
            print(f"[MPC] QP warning: {result.message}")
        u_opt = np.clip(result.x.reshape(N, NU)[0], U_LB, U_UB)
        self._last_u = u_opt.copy()
        return u_opt

    def torques_to_attitude(self, u_opt: np.ndarray):
        tau_x, tau_y, tau_z, nu_z = u_opt
        nu_z_safe = max(nu_z, NU_Z_HOV * 0.5)
        roll  = float(np.clip( tau_x / (nu_z_safe/4 * AL), -0.40,  0.40))
        pitch = float(np.clip(-tau_y / (nu_z_safe/4 * AL), -0.35,  0.35))
        thrust = float(np.clip((nu_z / (MASS * G)) * 0.5, 0.30, 0.85))
        return roll, pitch, thrust