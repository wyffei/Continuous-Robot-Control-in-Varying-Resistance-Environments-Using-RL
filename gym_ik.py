import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import os
import json
import random
from hydro_forces1104 import HydroForces
from pid_controller_twofluid import PIDController
from datetime import datetime
from fluid_field1104 import FluidField
from qp_ik import ConstraintIK
import multiprocessing as mp


class gymenv(gym.Env):
    """
    Pure IK baseline environment.

    Key characteristics:
    - The environment NO LONGER uses actions from PPO to control the arm.
      Instead, step() internally uses an IK controller to drive the end-effector
      straight toward a fixed target point.
    - Uses `ConstraintIK` (qp_ik.py) to convert a desired end-effector
      velocity dx_des into joint velocities qdot.
    - Still keeps the full physics: HydroForces (two-fluid hydrodynamics),
      PIDController, and MuJoCo dynamics.
    - No reinforcement learning reward is designed here: step() always
      returns reward = 0.0.
    - The main outputs are:
        * epi_energy: total energy consumption over the whole episode,
        * step_energy: energy consumed in the current step,
        * ee_distance: distance between end-effector and target,
        * is_success flag: whether the target is reached.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        xml_path,
        target_pos=None,
        render=True,
        log_path="work_dirs",
        max_steps=3000,
        segment_seconds=0.1,
    ):
        super().__init__()

        # ====== Load MuJoCo model and create data (same as RL env) ======
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Number of joints and initial joint positions (all zeros)
        init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.num_joints = self.model.nq
        for i, q in enumerate(init_qpos):
            addr = self.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i + 1}"
                )
            ]
            self.data.qpos[addr] = q

        # ====== Target position (fixed for IK baseline) ======
        # By default we fix the target at (0, 0, 0.7).
        if target_pos is None:
            self.target_pos = np.array(
                [
                    0.0,
                    0.0,
                    0.7,
                ]
            )
        else:
            self.target_pos = np.asarray(target_pos, dtype=float)

        self.render = render
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)

        # ====== Tip and goal marker for distance calculation and visualization ======
        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_site"
        )
        try:
            self.goal_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker"
            )
        except Exception:
            self.goal_geom_id = None

        # Move the goal marker to the chosen target position and forward the model
        if self.goal_geom_id is not None:
            self.model.geom_pos[self.goal_geom_id] = self.target_pos
            mujoco.mj_forward(self.model, self.data)

        if mp.current_process().name == "MainProcess":
            print(f"[IK Env] Model loaded, number of joints: {self.num_joints}")

        self.step_count = 0
        self.epi_energy = 0.0
        self.max_steps = max_steps  # Maximum number of physics steps per episode

        # ====== Two-phase fluid field and hydrodynamics (same as RL env) ======
        self.fluid_field = FluidField(
            single_fluid=False,
        )
        self.hydro = HydroForces(
            self.model,
            vc=np.array([0.0, 0.0]),
            a=0.05,
            b=0.04,
            l_half=0.075,
            field=self.fluid_field,
        )

        # ====== PID controller (same gains as RL environment) ======
        self.controller = PIDController(
            kp=[1300, 800, 600, 600, 500, 350, 300, 300, 150, 50][: self.num_joints],
            ki=[0.0] * self.num_joints,
            kd=[10, 11, 11, 11, 11, 9, 6, 3, 2, 0.2][: self.num_joints],
            Pmax=20000,
            derivative_filter_alpha=0.91,
            num_joints=self.num_joints,
            epsilon=1e-6,
            tau_max=500,
        )

        # Viewer (passive) only if render=True
        self.viewer = (
            mujoco.viewer.launch_passive(self.model, self.data) if render else None
        )
        self.segment_seconds = segment_seconds
        self.dt = float(self.model.opt.timestep)

        # ====== Termination threshold (same as RL env) ======
        self.termination_distance = 0.05  # success if distance < 5 cm

        # ====== IK configuration: NO velocity limits here ======
        # We remove explicit max_tip_speed, and also pass vel_limits=None to IK.
        # This means the IK will not clip joint velocities internally.
        self.ik = ConstraintIK(
            self.model,
            num_joints=self.num_joints,
            damp=1e-3,
            vel_limits=None,   # No joint velocity clipping inside IK
            q_limits=None,
            null_weight=1e-2,
        )

        # ====== Action space: dummy action (ignored in step()) ======
        # PPO still expects an action space, so we define a trivial one and
        # completely ignore the `action` argument inside step().
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=0.0,
            shape=(1,),
            dtype=np.float32,
        )

        # ====== Observation space: consistent with RL env ======
        sample_obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32,
        )

        # Optional: track best-energy successful trajectories
        self.best_success_energy = np.inf
        self.best_success_traj = None
        self.best_success_traj_ctrl = None
        self.success_counter = 0  # counter for consecutive successes

        self.record_enabled = False
        self.traj_record = []
        self.traj_record_ctrl = []
        self.total_sim_steps = 0

    # ------------------------------------------------------------------
    # Utility: seeding and observation construction
    # ------------------------------------------------------------------
    def seed(self, seed=None):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _get_obs(self):
        """
        Build the observation vector:
        - joint positions and velocities,
        - end-effector position,
        - error vector (target - ee),
        - remaining time budget.
        """
        qpos = self.data.qpos[: self.num_joints]
        qvel = self.data.qvel[: self.num_joints]
        ee_pos = self.data.site_xpos[self.tip_site_id].copy()
        ee_error = self.target_pos - ee_pos
        time_to_go = max(0.0, (self.max_steps - self.step_count) * self.dt)

        obs = np.concatenate(
            [
                qpos,
                qvel,
                ee_pos,
                ee_error,
                np.array([time_to_go], dtype=np.float32),
            ]
        ).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Core step: IK decides joint velocity; PPO action is ignored.
    # ------------------------------------------------------------------
    def step(self, action):
        """
        One simulation step.

        Notes:
        - The incoming `action` from PPO is COMPLETELY IGNORED.
        - Instead, we compute a desired end-effector velocity dx_des that points
          directly toward the fixed target, then use IK to compute joint
          velocities qdot.
        - We run physics for `segment_seconds` and accumulate:
            * hydrodynamic forces,
            * PID control torques,
            * energy consumption (epi_energy).
        - The reward is always 0.0, since this environment is purely
          for baseline evaluation (not for learning).
        """

        self_collision = False

        # ====== 1) Desired end-effector velocity dx_des ======
        ee_pos = self.data.site_xpos[self.tip_site_id].copy()
        vec = self.target_pos - ee_pos
        dist = float(np.linalg.norm(vec))

        if dist < 1e-6:
            # If already extremely close to target: zero desired velocity
            dx_des = np.zeros(3, dtype=float)
        else:
            # No explicit speed limit: velocity magnitude is proportional to distance.
            # The gain k can be tuned for faster or slower motion.
            k = 1.13
            dx_des = k * vec  # direction toward target, scaled by k

        # Current joint angles
        q = self.data.qpos[: self.num_joints].copy()

        # IK: convert end-effector velocity dx_des to joint velocity qdot
        qdot = self.ik.solve(
            data=self.data,
            dx_des=dx_des,
            q=q,
            q_mid=None,
        )

        # ====== 2) Physics integration loop ======
        model, data = self.model, self.data
        nj, dt = self.num_joints, self.dt

        # Number of small MuJoCo steps per "segment_seconds" of wall-clock time
        steps = max(2, int(self.segment_seconds / max(self.dt, 1e-6)))

        q_target = data.qpos[:nj].copy()
        total_step_energy = 0.0
        collision_penalty = 0.0

        for _ in range(steps):
            # 2-1) Integrate desired joint angles
            q_target += qdot * dt

            # 2-2) Compute hydrodynamic forces for all links
            data.qfrc_applied[:] = self.hydro.compute_qfrc_applied(data)

            # 2-3) PID control to track the target joint angles
            self.controller.step(model, data, current_des=q_target)

            # 2-4) Advance MuJoCo simulation by one step
            mujoco.mj_step(model, data)

            # 2-5) Self-collision detection
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                try:
                    geom1_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                    )
                    geom2_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                    )
                except Exception:
                    geom1_name = None
                    geom2_name = None

                if geom1_name is None or geom2_name is None:
                    continue

                if any(k in geom1_name for k in ["base", "segment"]) and any(
                    k in geom2_name for k in ["base", "segment"]
                ):
                    self_collision = True
                    collision_penalty = 8.0
                    break

            self.total_sim_steps += 1

            # 2-6) Energy accumulation: ∫ |τ · q̇| dt
            qvel = data.qvel[:nj]
            tau = data.ctrl[:nj]
            sub_energy = float(np.sum(np.abs(tau * qvel)) * dt)
            total_step_energy += sub_energy
            self.epi_energy += sub_energy

            self.step_count += 1
            if self.step_count >= self.max_steps:
                break

            if self.render and self.viewer is not None:
                self.viewer.sync()

            if self.record_enabled:
                self.traj_record.append(data.qpos.copy())
                self.traj_record_ctrl.append(q_target.copy())

        # ====== 3) Termination conditions ======
        ee_pos = data.site_xpos[self.tip_site_id].copy()
        dist = float(np.linalg.norm(self.target_pos - ee_pos))

        terminated = dist < self.termination_distance
        truncated = self.step_count >= self.max_steps

        if terminated:
            self.success_counter += 1
            print(
                f"[IK Env] Target reached! Steps: {self.step_count}, "
                f"Energy: {self.epi_energy:.4f}"
            )

            # Optionally record the best-energy trajectory
            if (
                self.record_enabled
                and len(self.traj_record) > 0
                and self.epi_energy < self.best_success_energy
            ):
                self.best_success_energy = self.epi_energy
                self.best_success_traj = np.array(self.traj_record)
                self.best_success_traj_ctrl = np.array(self.traj_record_ctrl)

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path = os.path.join(
                    self.log_path, f"best_traj_ik_only_{timestamp}.npz"
                )
                np.savez(
                    save_path,
                    traj=self.best_success_traj,
                    traj_ctrl=self.best_success_traj_ctrl,
                    total_steps=self.total_sim_steps,
                    epi_energy=self.epi_energy,
                )
                print(f"[IK Env] New best-energy episode saved to: {save_path}")

        if truncated:
            # Reset success counter if episode truncated
            self.success_counter = 0

        # ====== 4) Build observation and info dict ======
        obs = self._get_obs()
        reward = 0.0  # fixed reward for IK baseline (no RL objective here)

        info = {
            "reward": float(reward),               # kept for compatibility
            "epi_energy": float(self.epi_energy),  # total episode energy
            "step_energy": float(total_step_energy),
            "ee_distance": float(dist),
            "collision_penalty": float(collision_penalty),
            "is_success": bool(terminated),
        }

        return obs, reward, bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # reset(): reset the environment state and counters
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """
        Reset MuJoCo state, counters, energy, and the fixed target.

        The IK control logic is unchanged; we only reinitialize the
        environment for a new episode.
        """
        if seed is not None:
            self.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Fixed target position for IK baseline
        self.target_pos = np.array(
            [
                0.0,
                0.0,
                0.7,
            ]
        )
        if self.goal_geom_id is not None:
            self.model.geom_pos[self.goal_geom_id] = self.target_pos
            mujoco.mj_forward(self.model, self.data)

        # Reset counters and energy
        self.step_count = 0
        self.epi_energy = 0.0

        # Clear trajectory records
        self.traj_record = []
        self.traj_record_ctrl = []
        self.total_sim_steps = 0

        # Reset joint angles to zero configuration
        init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i, q in enumerate(init_qpos):
            addr = self.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i + 1}"
                )
            ]
            self.data.qpos[addr] = q

        obs = self._get_obs()
        return obs, {}
