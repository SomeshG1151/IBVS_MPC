import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve_discrete_are, block_diag, expm


MASS = 1.73
G = 9.81

H = 0.02
N = 25

NX = 6
NU = 4

TAU_RP = 0.35

ROLL_MAX = 0.45
PITCH_MAX = 0.40
YAW_RATE_MAX = 0.5

NU_Z_HOV = MASS * G

U_LB = np.array([-ROLL_MAX, -PITCH_MAX, -YAW_RATE_MAX, 0.0])
U_UB = np.array([ROLL_MAX, PITCH_MAX, YAW_RATE_MAX, 4 * G * MASS])


def _build_matrices():
    Ac = np.zeros((NX, NX))
    Bc = np.zeros((NX, NU))

    Ac[0, 4] = -G
    Ac[1, 3] = G
    Ac[3, 3] = -1.0 / TAU_RP
    Ac[4, 4] = -1.0 / TAU_RP

    Bc[3, 0] = 1.0 / TAU_RP
    Bc[4, 1] = 1.0 / TAU_RP
    Bc[5, 2] = 1.0
    Bc[2, 3] = -1.0 / MASS

    M = np.zeros((NX + NU, NX + NU))
    M[:NX, :NX] = Ac
    M[:NX, NX:] = Bc

    Md = expm(M * H)

    A = Md[:NX, :NX]
    B = Md[:NX, NX:]

    return A, B


class LinearMPC:
    def __init__(self):
        self.A, self.B = _build_matrices()

        self.Qx = np.diag([3.0, 3.0, 4.0, 1.0, 1.0, 0.05])
        self.Qu = np.diag([5.0, 5.0, 0.5, 8.0 / (NU_Z_HOV ** 2)])

        try:
            self.Qf = solve_discrete_are(self.A, self.B, self.Qx, self.Qu)
        except Exception:
            print("[MPC] Riccati failed, using 10*Qx")
            self.Qf = 10 * self.Qx

        self._last_u = np.array([0.0, 0.0, 0.0, NU_Z_HOV])

        self._x_star_filt = np.zeros(NX)
        self.X_STAR_ALPHA = 0.5

        self.Fx, self.Fu = self._build_condensed_prediction()

        self.Q_bar = block_diag(*([self.Qx] * N + [self.Qf]))
        self.R_bar = block_diag(*([self.Qu] * N))

        H_mat = self.Fu.T @ self.Q_bar @ self.Fu + self.R_bar
        H_mat = 0.5 * (H_mat + H_mat.T)

        rate_gains = np.array([14.0, 12.0, 3.0, 0.05])
        self.rate_weights = np.tile(rate_gains, N)
        self.H_mat = H_mat + np.diag(self.rate_weights)

        self.P = sparse.csc_matrix(self.H_mat)
        self.A_box = sparse.eye(N * NU, format="csc")

        u_eq = np.array([0.0, 0.0, 0.0, NU_Z_HOV])
        dU_LB = np.tile(U_LB - u_eq, N)
        dU_UB = np.tile(U_UB - u_eq, N)

        self._osqp = osqp.OSQP()
        self._osqp.setup(
            P=self.P,
            q=np.zeros(N * NU),
            A=self.A_box,
            l=dU_LB,
            u=dU_UB,
            verbose=False,
            warm_start=True,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=4000,
        )

        print(f"[MPC] ROLL_MAX={ROLL_MAX:.3f} rad  PITCH_MAX={PITCH_MAX:.3f} rad  NU_Z_HOV={NU_Z_HOV:.2f} N")
        print(f"[MPC] A norm={np.linalg.norm(self.A):.3f}  B norm={np.linalg.norm(self.B):.3f}")
        print(f"[MPC] TAU_RP={TAU_RP}s  X_STAR_ALPHA={self.X_STAR_ALPHA}")

    def compute_reference(self, vc, roll=0.0, pitch=0.0, yaw=0.0):
        R_Bc = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        v_body = R_Bc @ vc[:3]

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

        R_body_to_inertial = Rz @ Ry @ Rx
        v_inertial = np.clip(R_body_to_inertial @ v_body, -1.0, 1.0)

        x_star = np.zeros(NX)
        x_star[:3] = v_inertial

        self._x_star_filt = self.X_STAR_ALPHA * x_star + (1 - self.X_STAR_ALPHA) * self._x_star_filt
        return self._x_star_filt.copy()

    def _build_condensed_prediction(self):
        Fx = np.zeros(((N + 1) * NX, NX))
        Fu = np.zeros(((N + 1) * NX, N * NU))

        Ak = np.eye(NX)
        for k in range(N + 1):
            Fx[k * NX:(k + 1) * NX] = Ak
            Ak = self.A @ Ak

        for k in range(1, N + 1):
            for j in range(k):
                Fu[k * NX:(k + 1) * NX, j * NU:(j + 1) * NU] = np.linalg.matrix_power(self.A, k - 1 - j) @ self.B

        return Fx, Fu

    def solve(self, x_current, x_star):
        u_eq = np.array([0.0, 0.0, 0.0, NU_Z_HOV])

        x_tilde = x_current - x_star
        f = self.Fu.T @ self.Q_bar @ self.Fx @ x_tilde

        du_last = np.tile(self._last_u - u_eq, N)
        f_reg = f - self.rate_weights * du_last

        dU_LB = np.tile(U_LB - u_eq, N)
        dU_UB = np.tile(U_UB - u_eq, N)

        self._osqp.update(q=f_reg, l=dU_LB, u=dU_UB)
        result = self._osqp.solve()

        if result.info.status in ("solved", "solved inaccurate"):
            du_opt = result.x.reshape(N, NU)[0]
        else:
            print(f"[MPC] QP warning: {result.info.status}")
            du_opt = du_last[:NU]

        u_opt = np.clip(du_opt + u_eq, U_LB, U_UB)
        self._last_u = u_opt.copy()

        return u_opt

    def to_attitude_command(self, u_opt):
        roll_cmd, pitch_cmd, _, nu_z = u_opt

        roll = float(np.clip(roll_cmd, -ROLL_MAX, ROLL_MAX))
        pitch = float(np.clip(pitch_cmd, -PITCH_MAX, PITCH_MAX))
        thrust = float(np.clip((nu_z / (MASS * G)) * 0.5, 0.30, 0.85))

        return roll, pitch, thrust