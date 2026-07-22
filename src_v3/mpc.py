import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve_discrete_are, block_diag, expm

MASS = 1.73
GRAVITY = 9.81

TIMESTEP   = 0.02
HORIZON    = 25

NX = 6
NU = 4

ROLL_TIME_CONSTANT = 0.15

ROLL_LIMIT      = 0.7
PITCH_LIMIT     = 0.65
YAW_RATE_LIMIT  = 0.5

HOVER_THRUST = MASS * GRAVITY

CONTROL_LB = np.array([-ROLL_LIMIT, -PITCH_LIMIT, -YAW_RATE_LIMIT, 0.0])
CONTROL_UB = np.array([ ROLL_LIMIT,  PITCH_LIMIT,  YAW_RATE_LIMIT, 4 * GRAVITY * MASS])

NU_Z_HOV = HOVER_THRUST


def _discretize_dynamics():
    Ac = np.zeros((NX, NX))
    Bc = np.zeros((NX, NU))

    Ac[0, 4] = -GRAVITY
    Ac[1, 3] =  GRAVITY
    Ac[3, 3] = -1.0 / ROLL_TIME_CONSTANT
    Ac[4, 4] = -1.0 / ROLL_TIME_CONSTANT

    Bc[3, 0] =  1.0 / ROLL_TIME_CONSTANT
    Bc[4, 1] =  1.0 / ROLL_TIME_CONSTANT
    Bc[5, 2] =  1.0
    Bc[2, 3] = -1.0 / MASS

    M = np.zeros((NX + NU, NX + NU))
    M[:NX, :NX] = Ac
    M[:NX, NX:] = Bc

    Md = expm(M * TIMESTEP)
    return Md[:NX, :NX], Md[:NX, NX:]


class LinearMPC:
    def __init__(self):
        self.A, self.B = _discretize_dynamics()

        self.state_cost = np.diag([3.0, 3.0, 4.0, 1.0, 1.0, 0.05])
        self.input_cost = np.diag([3.0, 3.0, 0.5, 8.0 / (HOVER_THRUST ** 2)])

        try:
            self.terminal_cost = solve_discrete_are(self.A, self.B, self.state_cost, self.input_cost)
        except Exception:
            print("[MPC] Riccati failed, falling back to 10×state_cost")
            self.terminal_cost = 10 * self.state_cost

        self._last_input       = np.array([0.0, 0.0, 0.0, HOVER_THRUST])
        self._reference_filter = np.zeros(NX)
        self.REFERENCE_ALPHA   = 0.5

        self.Fx, self.Fu = self._build_prediction_matrices()

        Q_bar = block_diag(*([self.state_cost] * HORIZON + [self.terminal_cost]))
        R_bar = block_diag(*([self.input_cost] * HORIZON))

        H = self.Fu.T @ Q_bar @ self.Fu + R_bar
        H = 0.5 * (H + H.T)

        rate_gains        = np.array([14.0, 12.0, 3.0, 0.05])
        self.rate_weights = np.tile(rate_gains, HORIZON)
        self.H_mat        = H + np.diag(self.rate_weights)
        self.Q_bar        = Q_bar

        equilibrium   = np.array([0.0, 0.0, 0.0, HOVER_THRUST])
        delta_lb      = np.tile(CONTROL_LB - equilibrium, HORIZON)
        delta_ub      = np.tile(CONTROL_UB - equilibrium, HORIZON)

        self._solver = osqp.OSQP()
        self._solver.setup(
            P=sparse.csc_matrix(self.H_mat),
            q=np.zeros(HORIZON * NU),
            A=sparse.eye(HORIZON * NU, format="csc"),
            l=delta_lb,
            u=delta_ub,
            verbose=False,
            warm_start=True,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=4000,
        )

    def _build_prediction_matrices(self):
        Fx = np.zeros(((HORIZON + 1) * NX, NX))
        Fu = np.zeros(((HORIZON + 1) * NX, HORIZON * NU))

        Ak = np.eye(NX)
        for k in range(HORIZON + 1):
            Fx[k*NX:(k+1)*NX] = Ak
            Ak = self.A @ Ak

        for k in range(1, HORIZON + 1):
            for j in range(k):
                Fu[k*NX:(k+1)*NX, j*NU:(j+1)*NU] = (
                    np.linalg.matrix_power(self.A, k - 1 - j) @ self.B
                )

        return Fx, Fu

    def compute_reference(self, camera_velocity, roll=0.0, pitch=0.0, yaw=0.0):
        R_cam_to_body = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        v_body        = R_cam_to_body @ camera_velocity[:3]

        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)

        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

        v_inertial = np.clip((Rz @ Ry @ Rx) @ v_body, -5, 5)

        reference       = np.zeros(NX)
        reference[:3]   = np.clip(v_inertial, -3, 3)

        self._reference_filter = (
            self.REFERENCE_ALPHA * reference
            + (1 - self.REFERENCE_ALPHA) * self._reference_filter
        )
        return self._reference_filter.copy()

    def solve(self, current_state, reference_state):
        equilibrium = np.array([0.0, 0.0, 0.0, HOVER_THRUST])
        error       = current_state - reference_state

        gradient       = self.Fu.T @ self.Q_bar @ self.Fx @ error
        delta_last     = np.tile(self._last_input - equilibrium, HORIZON)
        gradient_reg   = gradient - self.rate_weights * delta_last

        delta_lb = np.tile(CONTROL_LB - equilibrium, HORIZON)
        delta_ub = np.tile(CONTROL_UB - equilibrium, HORIZON)

        self._solver.update(q=gradient_reg, l=delta_lb, u=delta_ub)
        result = self._solver.solve()

        if result.info.status in ("solved", "solved inaccurate"):
            best_delta = result.x.reshape(HORIZON, NU)[0]
        else:
            print(f"[MPC] QP solver: {result.info.status}")
            best_delta = delta_last[:NU]

        optimal_input  = np.clip(best_delta + equilibrium, CONTROL_LB, CONTROL_UB)
        self._last_input = optimal_input.copy()
        return optimal_input

    def to_attitude_command(self, optimal_input):
        roll_cmd, pitch_cmd, _, thrust_force = optimal_input

        roll   = float(np.clip(roll_cmd,   -ROLL_LIMIT,  ROLL_LIMIT))
        pitch  = float(np.clip(pitch_cmd,  -PITCH_LIMIT, PITCH_LIMIT))
        thrust = float(np.clip((thrust_force / (MASS * GRAVITY)) * 0.5, 0.30, 0.85))

        return roll, pitch, thrust