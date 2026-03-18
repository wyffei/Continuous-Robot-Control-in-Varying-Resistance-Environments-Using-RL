"""
hydro_forces.py

Hydrodynamic force module for MuJoCo links based on:
Kelasidi et al., "Modeling of underwater snake robots" (ICRA 2014).
(uses Morison-like decomposition: added-mass + linear drag + nonlinear drag,
 plus rotational drag torque terms lambda1/2/3)

Usage:
    hydro = HydroForces(model, data, per_body_params=..., vc=np.array([0.0,0.0]))
    qfrc_hydro = hydro.compute_qfrc_applied(data)
    # or: hydro.apply_to_data(data)  # which adds into data.qfrc_applied in-place
"""
import numpy as np
import mujoco

class HydroForces:
    def __init__(self, model, vc=np.array([0.0, 0.0]),
                 a=0.05, b=0.04, l_half=0.075,
                 rho=1000.0, Cf=0.03, CD=2.0, CA=1.0, CM=1.0,
                 field=None, g=9.80): 
        
        self.model = model
        self.vc = np.asarray(vc, dtype=float)
        self.g = g
        
        # default geometric params (scalars or per-body arrays)
        self.a = a
        self.b = b
        self.l = l_half  # half-length
        self.rho = rho
        self.Cf = Cf
        self.CD = CD
        self.CA = CA
        self.CM = CM
        self.field = field

        # precompute per-link coefficients for convenience (scalars or arrays)
        # ĈD = diag[ ct, cn ]  (paper eq. 14)
        # ct = 1/2 * rho * pi * Cf * (b + a) / (2l)
        # cn = 1/2 * rho * CD * 2a * 2l  -> but paper gives form; we follow numeric forms used in paper
        # We'll compute ct, cn following paper's provided expressions that yield consistent units.
        # Use paper's scalar expressions (paper section C / numeric example).
        a = self.a
        b = self.b
        l = self.l
        rho = self.rho
        Cf = self.Cf
        CD = self.CD

        # tangential viscous coefficient (ct) derived from Cf (axial)
        # Using paper eq (14) style: ct = 1/2 * rho * pi * Cf * (b + a) / (2l)
        self.ct = 0.5 * rho * np.pi * Cf * ((b + a) / 2) * 2*l
        # normal drag (cn) from CD: cn = 1/2 * rho * CD * (2a) * (2l) / (area normalization)
        # paper's numeric ĈD in example leads to ct ~0.2639 and cn ~8.4 with their numbers.
        # To replicate the paper's numeric scale use:
        self.cn = 0.5 * rho * CD * (2.0 * a) * (2.0 * l)   # simplified -> rho * CD * a
        # The above choice keeps units consistent; numeric tuning may be needed by user.

        # Added mass matrix diag[mu_t, mu_n] and torque inertia params
        # paper gives ĈA diag [0, rho*pi*CA*a^2*2l] / normalization:
        self.mu_t = 0
        self.mu_n = rho * np.pi * self.CA * (a**2) * (2.0 * l)  # matches eq. (15) scale

        # rotational torque parameters lambda1,2,3 (paper eqs 17 & 20)
        CM = self.CM
        self.lambda1 = (1.0 / 12.0) * rho * np.pi * CM * ((a**2 - b**2)**2.0) *(l**3)
        self.lambda2 = (1.0 / 6.0) * rho * np.pi * Cf * (a + b) * (l**3)
        self.lambda3 = (1.0 / 8.0) * rho * np.pi * Cf * (a + b) * (l**4)

        # internal state to approximate accelerations by finite difference
        # arrays indexed by body id: store previous velocity (linear 3) and angular 3, and prev theta_dot (z)
        self.prev_linvel = {}
        self.a_filtered={}
        self.prev_angvel_body = {}
        self.prev_theta_dot = {}
        self.prev_com_vel= {}
        self.initialized = False

        # convenience
        self.nv = model.nv
        self.nbody = model.nbody
        # We'll only process bodies that have positive mass (skip static world/body with mass 0)
        # Build a list of body ids to process
        self.body_ids = [i for i in range(1, self.nbody) if (hasattr(model, "body_mass") and model.body_mass[i] > 0) or (not hasattr(model, "body_mass"))]
        # if model.body_mass not present or zeros, fall back to all bodies except 0
        if len(self.body_ids) == 0:
            self.body_ids = list(range(1, self.nbody))

    def _init_state(self, data):
        # initialize previous velocities to current ones
        for bid in self.body_ids:
            lin = np.array(data.cvel[bid][3:], dtype=float)
            ang = np.array(data.cvel[bid][:3], dtype=float)
            self.prev_linvel[bid] =0
            self.prev_com_vel[bid] = np.array(data.xipos[bid], dtype=float) 
            # angular velocity in body frame
            R = data.xmat[bid].reshape(3,3)
            ang_body = R.T @ ang
            # ang_body =  ang
            self.prev_angvel_body[bid] = ang_body.copy()
            self.prev_theta_dot[bid] = ang_body[1]  # z-axis
        self.initialized = True



    def get_body_com_global_vel(self,model, data, body_id):
        """
        Returns the body's center of mass linear and angular velocities in the global coordinate system
        :param model: mujoco.MjModel
        :param data: mujoco.MjData
        :param body_id: int, index of the body
        :return:
            v_com_global: ndarray (3,) linear velocity
            w_com_global: ndarray (3,) angular velocity
        """
        nv = model.nv

        # Allocate Jacobian matrices
        jacp = np.zeros((3, nv))  # Linear velocity Jacobian
        jacr = np.zeros((3, nv))  # Angular velocity Jacobian
        # Get body COM Jacobian
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)

        # Linear velocity = jacp @ qvel
        v_com_global = jacp @ data.qvel  # shape (3,)
        # Angular velocity = jacr @ qvel
        w_com_global = jacr @ data.qvel  # shape (3,)

        return v_com_global, w_com_global
    


    def get_body_com_acc(self,model, data, body_id):
        """
        Uses qacc + jacp to compute body COM linear and angular accelerations (global coordinate system)
        """
        nv = model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))

        # Jacobian matrices
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)

        # Compute acceleration using qacc
        # Note: This only considers J * qacc, and does not include jac_dot * qvel,
        # but is more stable than finite-difference when the conventional simulation dt is small and qvel is smooth.
        a_com = jacp @ data.qacc
        alpha_com = jacr @ data.qacc

        return a_com, alpha_com


    def _params_at_body(self, data, bid):
        if self.field is None:
            return dict(rho=self.rho, Cf=self.Cf, CD=self.CD, CA=self.CA, CM=self.CM)
        pos = np.array(data.xipos[bid], dtype=float)
        return self.field.at(pos)

    def compute_qfrc_applied(self, data):
        """
        Compute hydrodynamic generalized forces (qfrc) for all considered bodies.
        Returns a numpy array of length model.nv that can be added to data.qfrc_applied.
        Must be called every control step BEFORE mujoco.mj_step (so data.cvel corresponds to current state).
        """
        model = self.model
        dt = float(model.opt.timestep) if hasattr(model, "opt") else 1e-3
        if dt <= 0:
            dt = 1e-3

        if not self.initialized:
            self._init_state(data)

        nq = self.nv
        qfrc_total = np.zeros(nq, dtype=float)

        # iterate each body (link)
        for bid in self.body_ids:
            # Local fluid params
            prm = self._params_at_body(data, bid)
            rho, Cf, CD, CA, CM = prm["rho"], prm["Cf"], prm["CD"], prm["CA"], prm["CM"]
            
            a, b, l = self.a, self.b, self.l
            ct = 0.5 * rho * np.pi * Cf * ((b + a) / 2) * 2*l
            cn = 0.5 * rho * CD * (2.0 * a) * (2.0 * l)
            mu_t = rho * np.pi * CA * (a**2) * (2.0 * l)/5
            mu_n = rho * np.pi * CA * (a**2) * (2.0 * l)
            lam1 = (1.0/12.0) * rho * np.pi * CM * ((a**2 - b**2)**2.0) * (l**3)
            lam2 = (1.0/6.0) * rho * np.pi * Cf * (a + b) * (l**3)
            lam3 = (1.0/8.0) * rho * np.pi * Cf * (a + b) * (l**4)

            # rotation matrix from body -> global
            R = data.xmat[bid].reshape(3,3)

            # global position (for gravity / interface)
            pos = np.array(data.xipos[bid], dtype=float)
            z_pos = pos[2]  # z>0: air, z<=0: water

            # global velocities
            v_global, w_global = self.get_body_com_global_vel(model, data, bid)

            # to body frame
            v_body = R.T @ v_global
            w_body = R.T @ w_global

            # finite-diff accel in body frame
            prev_v = self.prev_linvel.get(bid, np.zeros(3))
            prev_a = self.a_filtered.get(bid, np.zeros(3))
            # a_body = (v_body - prev_v) / dt
            a_body_raw = (v_body - prev_v) / dt
            alpha = 0.8  # Smoothing factor (smaller means smoother)
            a_body = alpha * a_body_raw + (1-alpha) * prev_a

            prev_wb = self.prev_angvel_body.get(bid, np.zeros(3))
            alpha_body = (w_body - prev_wb) / dt

            # Changed to xz plane / rotation about y-axis
            # 1) Rotation axis changed to y:
            theta_dot  = w_body[1] 
            theta_ddot = (theta_dot - self.prev_theta_dot.get(bid, 0.0)) / dt

            # 2) Current (background flow) in xz plane: vc = (vc_x, 0, vc_z)
            vc_body = R.T @ np.array([self.vc[0], 0.0, self.vc[1]]) 

            # 3) Relative velocity/acceleration takes the (x, z) two components
            vrel = np.array([v_body[0] - vc_body[0], v_body[2] - vc_body[2]])  
            arel = np.array([a_body[0],              a_body[2]])               

            # -- Added mass force (body frame, 2D in xz) --
            fA_body_2 = - np.array([mu_t * arel[0], mu_n * arel[1]])
            Fmax = 500
            fA_body_2 = np.clip(fA_body_2, -Fmax, Fmax)

            # -- Linear drag (body frame, 2D in xz) --
            fD_lin_body_2 = - np.array([ct * vrel[0], cn * vrel[1]])

            # -- Quadratic drag (body frame, 2D in xz) --
            fD_quad_body_2 = - np.array([ct * np.sign(vrel[0]) * (vrel[0]**2),
                                        cn * np.sign(vrel[1]) * (vrel[1]**2)])

            # total hydro force in body xz-plane
            f_body_2 = fA_body_2 + fD_lin_body_2 + fD_quad_body_2

            # 4) Embed back into 3D: currently in xz plane, so y=0
            f_body_3 = np.array([f_body_2[0], 0.0, f_body_2[1]], dtype=float) 

            # to global
            f_global = R @ f_body_3

            # gravity in global
            gravity_force = np.zeros(3)
            if z_pos > 0:
                mass = model.body_mass[bid] if hasattr(model, "body_mass") else 1.0
                gravity_force = np.array([0.0, 0.0, -mass * self.g])

            total_force = f_global + gravity_force

            # Rotational fluid torque about CM (around y-axis）
            tau_y = - (lam1 * theta_ddot + lam2 * theta_dot + lam3 * theta_dot * abs(theta_dot))
            torque_body_3 = np.array([0.0, tau_y, 0.0], dtype=float)   
            torque_global = R @ torque_body_3

            # apply force & torque
            qfrc_contrib = np.zeros(nq, dtype=float)
            mujoco.mj_applyFT(self.model, data,
                            np.ascontiguousarray(total_force.astype(np.float64)),
                            np.ascontiguousarray(torque_global.astype(np.float64)),
                            np.ascontiguousarray(pos.astype(np.float64)),
                            int(bid),
                            qfrc_contrib)
            qfrc_total += qfrc_contrib

            # update caches
            self.prev_linvel[bid] = v_body.copy()
            self.prev_angvel_body[bid] = w_body.copy()
            self.prev_theta_dot[bid] = theta_dot
            self.a_filtered[bid]=a_body.copy()
        return qfrc_total


    def apply_to_data(self, data):
        """Compute & add hydrodynamic generalized forces to data.qfrc_applied in-place."""
        qfrc = self.compute_qfrc_applied(data)
        # ensure data.qfrc_applied exists and has correct length
        try:
            data.qfrc_applied[:] += qfrc
        except Exception:
            # fallback: set directly
            for i in range(min(len(data.qfrc_applied), len(qfrc))):
                data.qfrc_applied[i] += qfrc[i]
