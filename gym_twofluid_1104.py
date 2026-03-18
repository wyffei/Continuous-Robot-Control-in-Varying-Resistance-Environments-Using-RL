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
    Optimized Environment (gym_ik_clean.py):
    1. Removed all IK guidance and heuristic rules (anti-stagnation, automatic speed reduction, etc.).
    2. Simplified the action space to direct joint velocity control.
    3. Simplified the reward function to three clear goals: progress, energy consumption, success.
    """

    def __init__(self, xml_path, target_pos=None, render=True, log_path="work_dirs", max_steps=3000,
                 segment_seconds=0.1): 
        super().__init__()

        # Model Loading
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i, q in enumerate(init_qpos):
            addr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")]
            self.data.qpos[addr] = q
       
        self.target_pos = np.array([0, 0.0, 0.7])
        self.num_joints = self.model.nq
        self.q_target = init_qpos
        self.render = render
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        if mp.current_process().name == "MainProcess":
            print(f"[INFO] (Clean Env) Model detected number of joints: {self.num_joints}")
        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip_site")
        try:
            self.goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")
        except Exception:
            self.goal_geom_id = None
        if self.goal_geom_id is not None:
            self.model.geom_pos[self.goal_geom_id] = self.target_pos
            mujoco.mj_forward(self.model, self.data)  
        self.step_count = 0
        self.epi_energy = 0
        self.max_steps = max_steps # Physics steps

        self.fluid_field = FluidField(
            single_fluid=False,  # Enable two-phase flow
        )

        # Physics and Control
        self.hydro = HydroForces(
            self.model, vc=np.array([0.0, 0.0]), a=0.05, b=0.04, l_half=0.075,
            field=self.fluid_field
        )
        self.controller = PIDController(
            kp=[1300, 800, 600, 600, 500, 350, 300, 300, 150, 50][:self.num_joints],
            ki=[0.0] * self.num_joints,
            kd=[10, 11, 11, 11, 11, 9, 6, 3, 2, 0.2][:self.num_joints],
            Pmax=20000, derivative_filter_alpha=0.91, num_joints=self.num_joints,
            epsilon=1e-6, tau_max=500
        )
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if render else None
        self.segment_seconds = segment_seconds
        self.dt = self.model.opt.timestep

        # Reward function hyperparameters 
        # Centralized definition of all reward/penalty weights for easy tuning
        self.w_progress = 100.0  # Scaling coefficient for progress reward (potential field reward)
        self.w_energy = 1.5  # Weight for energy consumption penalty
        self.w_time = 0.01    # Time penalty per step (encourages efficiency)
        self.bonus_success = 20.0  # Large bonus for reaching the target (was 12.0)
        self.termination_distance = 0.05 # Success radius (was 0.1)
        
        # Action space
        # Change action space from (11,) [direction + magnitude] to (10,) direct velocity control
        self.base_max_qdot = 3.5 # Maximum joint velocity
        self.action_space = gym.spaces.Box(
            low=-self.base_max_qdot, 
            high=self.base_max_qdot, 
            shape=(self.num_joints,), # Directly control 10 joints
            dtype=np.float32
        )
        
        self.ik = ConstraintIK(self.model, self.num_joints, damp=1e-3)

        # Observation space
        sample_obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32
        )
        
        # Previous distance for progress reward
        self._prev_dist = None

        # Record best successful episode
        self.best_success_energy = np.inf    # Current lowest energy consumption
        self.best_success_traj = None          # Current best trajectory
        self.best_success_traj_ctrl = None
        self.success_counter = 0             # Counter for consecutive successes
        self.record_enabled = False          # Recording initially disabled
        self.recordnext_enabled = False
        self.traj_record = []                # Temporary storage for joint target angle trajectory of current episode
        self.traj_record_ctrl = []
        self.total_sim_steps = 0


    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def _get_obs(self):
        qpos = self.data.qpos[:self.num_joints]
        qvel = self.data.qvel[:self.num_joints]
        ee_pos = self.data.site_xpos[self.tip_site_id].copy()
        ee_error = self.target_pos - ee_pos
        time_to_go = max(0.0, (self.max_steps - self.step_count) * self.dt)
        obs = np.concatenate([
            qpos, qvel, ee_pos, ee_error, 
            np.array([time_to_go], dtype=np.float32)
            # fluid_zone
        ]).astype(np.float32)
        return obs

    def step(self, action):
        
        self_collision = False
        collision_penalty = 0.0
        qdot = np.clip(action, -self.base_max_qdot, self.base_max_qdot)

        # Physics Step
        model, data = self.model, self.data
        nj, dt = self.num_joints, self.dt
        steps = max(2, int(self.segment_seconds / max(self.dt, 1e-6))) 

        q_target = data.qpos[:nj].copy()
        total_energy = 0.0

        for sub in range(steps):
            # Target position is gradually accumulated within the loop
            q_target += qdot * dt 
            
            # Hydrodynamic forces and PID controller are called within the loop
            data.qfrc_applied[:] = self.hydro.compute_qfrc_applied(data)
            self.controller.step(model, data, current_des=q_target)
            
            # Physics integration
            mujoco.mj_step(model, data)
            # Detect self-collision
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                try:
                    geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                    geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                except Exception:
                    geom1_name = None
                    geom2_name = None

                # Some geoms don't have names, skip directly
                if geom1_name is None or geom2_name is None:
                    continue

                # Filter out non-arm objects like ground, target sphere
                if any(k in geom1_name for k in ["base", "segment"]) and \
                   any(k in geom2_name for k in ["base", "segment"]):
                    self_collision = True
                    break

            # If collision occurs, add penalty
            if self_collision:
                collision_penalty = 8.0  # Tunable, e.g., 5 or 10

            self.total_sim_steps += 1
            # Accumulate energy consumption
            qvel = data.qvel[:nj]
            tau = data.ctrl[:nj]
            sub_energy = float(np.sum(np.abs(tau * qvel)) * dt)
            total_energy += sub_energy

            self.step_count += 1
            if self.step_count >= self.max_steps:
                break
            if self.render and self.viewer is not None:
                self.viewer.sync()
            
            # Record trajectory
            if self.record_enabled:
                self.traj_record.append(data.qpos.copy())
                self.traj_record_ctrl.append(q_target.copy())


        # Reward Function
        ee_pos = data.site_xpos[self.tip_site_id].copy()
        dist = float(np.linalg.norm(self.target_pos - ee_pos))
        if self._prev_dist is None:
            self._prev_dist = dist
        # Goal 1: Progress Reward (Potential Field Method)
        # (Distance at previous time step - Distance at current time step)
        # Closer to the goal, higher the reward.
        delta_dist = float(self._prev_dist - dist)
        r_progress = self.w_progress * delta_dist 
        
        if dist < 1.5:
            r_progress *=  1.2
        if dist < 0.9:
            r_progress *=  1.2
        if dist < 0.2:
            r_progress *=  1.5
        if r_progress > 25:
            r_progress = 25
        
        self._prev_dist = dist # Update distance for the previous time step

        # Goal 2: Energy Consumption Penalty，controlled by w_energy weight
        energy_penalty = self.w_energy * total_energy
        self.epi_energy += total_energy

        # Goal 3: Terminal Reward
        terminated = dist < self.termination_distance
        success_bonus = 0.0
        if terminated:
            success_bonus = self.bonus_success
        
        # Goal 4: Time Penalty (Encourages efficiency)
        time_reward = self.w_time

        # Goal 5: Joint Penalty
        angle_reward = np.abs(data.qpos[9].copy())+np.abs(data.qpos[8].copy())
        
        truncated = self.step_count >= self.max_steps
        if terminated or truncated:
            reward = (
            r_progress + 
            success_bonus - 
            collision_penalty-
            0.1*self.epi_energy-
            1.2*angle_reward-
            energy_penalty +
            time_reward
        )
        # Total Reward
        else:
            reward = (
                r_progress + 
                success_bonus - 
                collision_penalty-
                1.2*angle_reward-
                energy_penalty + 
                time_reward
            )

        # Check if successful episode trajectory should be saved
        if terminated :
            # Accumulate consecutive success count
            self.success_counter += 1
            print(f"[INFO] Successfully reached target! Current consecutive success count: {self.success_counter}")

            # Enable trajectory recording after 3 consecutive successes
            if not self.record_enabled and self.success_counter >= 3:
                self.record_enabled = True
                print("[INFO] Three consecutive successes, trajectory recording enabled.")
            
            # If recording is enabled, check if this is the new optimal energy consumption
            if self.record_enabled and len(self.traj_record) > 0 and self.epi_energy < self.best_success_energy:
                self.best_success_energy = self.epi_energy
                self.best_success_traj = np.array(self.traj_record)
                self.best_success_traj_ctrl = np.array(self.traj_record_ctrl)
                
                # Save to file
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_path = os.path.join(self.log_path, f"best_traj_{timestamp}.npy")
                np.savez(
                    save_path,
                    traj=self.best_success_traj,
                    traj_ctrl = self.best_success_traj_ctrl,
                    total_steps=self.total_sim_steps,
                    epi_energy = self.epi_energy
                )
                print(f"[INFO] New lowest energy consumption successful episode saved: {save_path} (Energy={self.epi_energy:.4f})")
        if truncated:
            # If failed or truncated, reset consecutive success count
            self.success_counter = 0

        # 4) Observation and Info
        obs = self._get_obs()
        info = {
            "reward_total": float(reward),
            "reward_progress": float(r_progress),
            "reward_success": float(success_bonus),
            "reward_energy": float(energy_penalty),
            "reward_time": float(time_reward),
            "epi_energy": float(self.epi_energy),
            "ee_distance": float(dist),
            "is_success": terminated, 
        }

        # (Optional) Record IK calculation direction, only for comparison/analysis, not part of control
        ik_dir = self.ik.solve(self.data, dx_des=self.target_pos - ee_pos, q=self.data.qpos[:self.num_joints])
        info["ik_alignment"] = float(np.dot(qdot / (np.linalg.norm(qdot)+1e-8), ik_dir))

        return obs, float(reward), bool(terminated), bool(truncated), info

    # --------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed) # Ensure random seed is set correctly

        mujoco.mj_resetData(self.model, self.data)

        self.target_pos = np.array([0, 0.0, 0.7]) 
        if self.goal_geom_id is not None:
            self.model.geom_pos[self.goal_geom_id] = self.target_pos
            mujoco.mj_forward(self.model, self.data)
        
        # Reset state
        self.step_count = 0
        self.epi_energy = 0
        
        # Clear trajectory cache
        self.traj_record = []
        self.traj_record_ctrl = []
        self.total_sim_steps = 0

        init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i, q in enumerate(init_qpos):
            addr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")]
            self.data.qpos[addr] = q
        # reset previous distance
        ee_pos = self.data.site_xpos[self.tip_site_id].copy()
        self._prev_dist = None
        
        obs = self._get_obs()
        return obs, {}