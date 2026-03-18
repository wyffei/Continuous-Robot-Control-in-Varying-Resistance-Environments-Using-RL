import numpy as np
import mujoco
import matplotlib.pyplot as plt
from openpyxl import Workbook
from hydro_forces1104 import HydroForces

class PIDController:
    def __init__(self, kp, ki, kd, Pmax=5.0, num_joints=10,
                 derivative_filter_alpha=0.9, integral_limit=10000000, tau_max=100, epsilon=1e-6):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.Pmax = Pmax
        self.num_joints = num_joints
        self.derivative_filter_alpha = derivative_filter_alpha
        self.integral_limit = integral_limit
        self.tau_max = tau_max
        self.epsilon = epsilon

        self.integral_error = np.zeros(num_joints)
        self.filtered_derror = np.zeros(num_joints)
        self.pre_des = np.zeros(num_joints)
        # Low-pass filter for tau_allowed
        self.tau_allowed_filtered = np.zeros(num_joints)
        self.tau_filter_alpha = 0
        self.dq_deadzone = 0.05

        self.time_log = []
        self.q_log = [[] for _ in range(num_joints)]
        self.des_log = [[] for _ in range(num_joints)]
        self.t = 0.0

    def export_angle_to_excel(self, filename="joint_angles.xlsx"):
        """
        Exports joint angle logs to Excel, each joint taking three columns: time, actual angle, desired angle
        """
        wb = Workbook()
        ws = wb.active
        ws.title = "Joint Angles"

        # Table header
        headers = []
        for j in range(self.num_joints):
            headers.append(f"Joint{j}_Time [s]")
            headers.append(f"Joint{j}_Angle [rad]")
            headers.append(f"Joint{j}_Desired [rad]")
        ws.append(headers)

        # Write data
        T = len(self.time_log)
        for i in range(T):
            row = []
            for j in range(self.num_joints):
                row.append(self.time_log[i])
                row.append(self.q_log[j][i])
                row.append(self.des_log[j][i])
            ws.append(row)

        wb.save(filename)
        print(f"Joint angle data exported to {filename}")

    def get_effective_gravity_qfrc(self, data):
        nq = self.nv
        qfrc_total_grav = np.zeros(nq, dtype=float)
        qfrc_g = np.zeros(self.model.nv)
        for bid in self.body_ids:
            z_pos = data.xipos[bid][2]
            if z_pos > 0:
                mass = self.model.body_mass[bid]
                f_grav = np.array([0, 0, -mass * self.g])
                mujoco.mj_applyFT(self.model, data, f_grav, np.zeros(3), data.xipos[bid], bid, qfrc_g)
                qfrc_total_grav += qfrc_g
        return qfrc_total_grav

    def step(self, model, data, current_des, mask=None):
        """
        mask: np.bool_[num_joints], True means PID is enabled for this joint, False means disabled 
              (might be controlled by external constant torque or other means)
        """
        dt = float(model.opt.timestep) if hasattr(model, "opt") else 1e-3
        nq = self.num_joints

        if mask is None:
            mask = np.ones(self.num_joints, dtype=bool)

        tau_pid = np.zeros(self.num_joints)
        tau_cmd = np.zeros(self.num_joints)
        qfrc_total_grav = np.zeros(nq, dtype=float)
        # Inner loop PID
        for i in range(self.num_joints):
            
            qfrc_g = np.zeros(nq)
            if not mask[i]:
                # Minimal maintenance for disabled PID joints to prevent D term jump on next enable
                self.pre_des[i] = current_des[i]
                continue
            # Calculate Gravity Feedforward
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")
            bid = model.jnt_bodyid[jid]  # joint -> body
            pos = np.array(data.xipos[bid])
            if pos[2] > 0:
                mass = model.body_mass[bid] if hasattr(model, "body_mass") else 1.0
                f_grav = np.array([0, 0, -mass * 9.8])
                qfrc_g = np.zeros(nq)
                mujoco.mj_applyFT(model, data,
                                f_grav,
                                np.zeros(3),
                                pos,
                                bid,
                                qfrc_g)
                qfrc_total_grav += qfrc_g

            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")
            qaddr = model.jnt_qposadr[jid]
            q = data.qpos[qaddr]
            dq = data.qvel[qaddr]

            error = current_des[i] - q
            raw_derror = 0 if self.t == 0 else - dq
            self.pre_des[i] = current_des[i]

            #Only integrate for enabled PID joints to prevent windup
            self.integral_error[i] += error * dt
            self.integral_error[i] = np.clip(self.integral_error[i], -self.integral_limit
                                             , self.integral_limit)

            # Low-pass filter for D term
            self.filtered_derror[i] = (
                self.derivative_filter_alpha * self.filtered_derror[i]
                + (1 - self.derivative_filter_alpha) * raw_derror
            )
            
            tau_pid[i] = (
                self.kp[i] * error
                + self.ki[i] * self.integral_error[i]
                + self.kd[i] * self.filtered_derror[i]
            )


        # Outer loop: Power limit + smoothing
        for i in range(self.num_joints):
            if not mask[i]:
                continue
         
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")
            qaddr = model.jnt_qposadr[jid]
            dq = data.qvel[qaddr]

            dq_eff = dq if abs(dq) > self.dq_deadzone else self.dq_deadzone
            tau_allowed_raw = self.Pmax / (abs(dq_eff) + self.epsilon)
            tau_allowed_raw = np.clip(tau_allowed_raw, -self.tau_max, self.tau_max)
            
            self.tau_allowed_filtered[i] = (
                self.tau_filter_alpha * self.tau_allowed_filtered[i]
                + (1 - self.tau_filter_alpha) * tau_allowed_raw
            )
        
        data.ctrl = tau_pid - qfrc_total_grav
        # Logging
        self.time_log.append(self.t)
        for i in range(self.num_joints):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")
            qaddr = model.jnt_qposadr[jid]
            self.q_log[i].append(data.qpos[qaddr])
            self.des_log[i].append(current_des[i])
        self.t += dt

    def plot_logs(self, joint_indices=[0]):
        for i in joint_indices:
            plt.figure()
            plt.plot(self.time_log, self.q_log[i], label=f"q{i+1} (actual)")
            plt.plot(self.time_log, self.des_log[i], '--', label=f"q{i+1} (desired)")
            plt.xlabel("Time [s]")
            plt.ylabel("Position [rad]")
            plt.title(f"Joint {i+1} tracking")
            plt.legend()
            plt.grid(True)
        plt.show()
