# qp_ik.py
import numpy as np
import mujoco


class ConstraintIK:
    """
    Lightweight, constraint-aware inverse kinematics for end-effector
    velocity control.

    Goal:
        Map a small desired end-effector displacement / velocity (dx_des)
        to joint velocities (qdot) for an n-DOF arm in MuJoCo.

    Method:
        - Uses Damped Least Squares (DLS) pseudo-inverse for the main task.
        - Adds a null-space term to keep joints away from limits / pull
          them toward a comfortable mid-configuration.
        - Soft joint limits and velocity limits are enforced via clipping,
          without calling an external QP solver.

    Design:
        This is a simple, self-contained approximation to a QP-based IK
        (no OSQP / qpOASES dependency). If needed, it can be replaced with a
        full QP solver in the future while keeping the same interface.
    """

    def __init__(
        self,
        model,
        num_joints,
        damp=1e-3,
        vel_limits=None,
        q_limits=None,
        null_weight=1e-2,
    ):
        """
        Args:
            model:      MuJoCo model.
            num_joints: Number of controllable joints (n).
            damp:       Damping factor for DLS (Tikhonov regularization).
            vel_limits: Optional [n] array of max joint velocities (rad/s).
                        If None, defaults to 8 rad/s for each joint.
            q_limits:   Optional (q_lo, q_hi) pair for joint angle limits.
                        If None, defaults to [-5, 5] rad for each joint.
            null_weight:Weight of null-space term to pull joints toward
                        mid-configuration.
        """
        self.model = model
        self.n = num_joints
        self.damp = float(damp)

        # Default joint velocity limits
        self.vel_limits = (
            vel_limits
            if vel_limits is not None
            else np.full(self.n, 8.0, dtype=float)
        )

        # Default joint position limits
        if q_limits is not None:
            self.q_lo, self.q_hi = q_limits
        else:
            self.q_lo = -np.full(self.n, 5.0, dtype=float)
            self.q_hi = np.full(self.n, 5.0, dtype=float)

        self.null_w = float(null_weight)

    def jac_site(self, data, site_name="tip_site"):
        """
        Compute the translational Jacobian (3 x n) of a MuJoCo site.

        Args:
            data:      MuJoCo data.
            site_name: Name of the site to compute Jacobian for.

        Returns:
            J: 3 x n positional Jacobian matrix of the site.
        """
        tip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        # mj_jacSite fills translational (jacp) and rotational (jacr) Jacobians
        mujoco.mj_jacSite(self.model, data, jacp, jacr, tip_id)

        # We only use the first n columns corresponding to controllable joints
        J = jacp[:, : self.n]
        return J

    def manipulability(self, J):
        """
        Compute Yoshikawa's translational manipulability measure.

        m = sqrt(det(J J^T))

        Args:
            J: 3 x n Jacobian matrix.

        Returns:
            scalar manipulability value.
        """
        JJt = J @ J.T
        return np.sqrt(np.maximum(1e-12, np.linalg.det(JJt)))

    def solve(self, data, dx_des, q, q_mid=None):
        """
        Solve IK for a small step in end-effector space.

        Args:
            data:   MuJoCo data.
            dx_des: Desired end-effector velocity / differential [3,].
            q:      Current joint configuration [n,].
            q_mid:  Preferred mid configuration in joint space [n,].
                    If None, defaults to the middle of joint limits.

        Returns:
            qdot: Joint velocity command [n,].
        """
        # 1) Compute Jacobian at current configuration
        J = self.jac_site(data)  # shape (3, n)
        JT = J.T

        # 2) Main task via Damped Least Squares:
        #    qdot_main = J^T (J J^T + λI)^(-1) dx_des
        A = J @ JT + self.damp * np.eye(3)
        qdot_main = JT @ np.linalg.solve(A, dx_des)

        # 3) Null-space objective to avoid joint limits and pull toward mid.
        if q_mid is None:
            # Default mid-configuration is the midpoint of joint limits
            q_mid = (self.q_lo + self.q_hi) * 0.5

        # "margin" from each limit, to weight joints near the limit more strongly
        margin = np.minimum(q - self.q_lo + 1e-3, self.q_hi - q + 1e-3)
        # Weight grows when margin becomes small (closer to limits)
        w = 1.0 / (margin**2)
        # Gradient-like term to pull q back toward q_mid
        z = -w * (q - q_mid)

        # 4) Construct null-space projection N = I - J^T (J J^T + λI)^(-1) J
        N = np.eye(self.n) - JT @ np.linalg.solve(A, J)
        qdot_null = N @ z

        # 5) Combine main task and null-space component
        qdot = qdot_main + self.null_w * qdot_null

        # 6) Clip joint velocities to [-vel_limits, vel_limits]
        qdot = np.clip(qdot, -self.vel_limits, self.vel_limits)
        return qdot
