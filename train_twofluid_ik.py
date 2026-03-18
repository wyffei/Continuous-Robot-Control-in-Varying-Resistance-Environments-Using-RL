"""
train_twofluid_ik.py
--------------------

This file trains an Imitation-Kinematics (IK)-based baseline controller
using RecurrentPPO, but **the PPO policy is ignored** because the
environment (gym_ik.py) internally executes IK-based control and
returns a fixed reward of 0.0.

The purpose of this file:
- Run the IK environment through a PPO loop to reuse episode logging.
- Collect episodic energy curves using a custom callback.
- Save checkpoints and logs for analysis.

The IK controller is the baseline used to compare against RL results.
"""

from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from sb3_contrib import RecurrentPPO
from reward_plot_callback import RewardPlotCallback
import pandas as pd
import os
from gym_ik import gymenv
from torch import nn
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
#  EpisodeEnergyPlotCallback
#  ------------------------------------------------------------
#  Records per-episode energy `epi_energy` from the environment,
#  and plots an energy curve at the end of each PPO rollout.
# ============================================================
class EpisodeEnergyPlotCallback(BaseCallback):
    """
    A callback to record and plot **episodic energy** returned from
    the IK environment. Unlike reward curves (unused in IK), this
    callback creates a real-time-updating PNG curve showing the total
    energy consumed in each episode.

    - Records epi_energy at episode termination.
    - Saves CSV periodically.
    - At each PPO rollout-end, redraws the entire energy curve.
    """

    def __init__(
        self,
        update_freq=1,          # Retained for compatibility; only controls logging density.
        save_interval=100,      # Number of steps after which CSV is saved.
        save_path="episode_energy.csv",
        plot_path=None,         # Optional: specify plot save path.
        verbose=0,
    ):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_interval = save_interval
        self.save_path = save_path
        self.plot_path = plot_path or os.path.splitext(save_path)[0] + ".png"

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        self.episode_energies = []
        self.episode_indices = []
        self.last_save_step = 0

    def _on_step(self):
        """
        Called at each environment step.
        Only logs epi_energy at episode ends.
        """
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        step = self.num_timesteps

        if dones is not None:
            for done, info in zip(dones, infos):
                if done and "epi_energy" in info:
                    epi_energy = float(info["epi_energy"])
                    self.episode_energies.append(epi_energy)
                    self.episode_indices.append(len(self.episode_energies))

        # Periodically save CSV file.
        if step - self.last_save_step >= self.save_interval:
            self._save_to_csv()
            self.last_save_step = step

        return True

    def _on_rollout_end(self):
        """
        Redraw energy curve at the end of each PPO rollout.
        Note:
        - Only saving the PNG file (no live GUI display).
        """
        if len(self.episode_energies) == 0:
            return True

        plt.ion()   # Enable interactive updates

        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()

        self.ax.plot(self.episode_indices, self.episode_energies,
                     lw=1.8, label="Episodic Energy (J)")

        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Energy (J)")
        self.ax.set_title("Episodic Energy Over Training")
        self.ax.grid(True)
        self.ax.legend()

        # Refresh GUI if available
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Save to PNG
        self.fig.savefig(self.plot_path, dpi=150, bbox_inches="tight")
        return True

    def _save_to_csv(self):
        """Save all recorded episodic energy to CSV."""
        df = pd.DataFrame({
            "Episode": self.episode_indices,
            "Epi_Energy": self.episode_energies
        })
        df.to_csv(self.save_path, index=False)
        if self.verbose:
            print(f"Saved energy table → {self.save_path}")

    def _on_training_end(self):
        """Save final plots and CSV on training completion."""
        self._save_to_csv()
        self._on_rollout_end()
        print(f"Final episodic energy saved to {self.save_path}, plot saved to {self.plot_path}")


# ============================================================
#  TrainLoggerCallback
#  ------------------------------------------------------------
#  Saves periodic PPO checkpoints and basic logs.
# ============================================================
class TrainLoggerCallback(BaseCallback):
    """
    Periodically saves PPO checkpoints for reproducibility.
    PPO policy is not used for control in IK mode, but checkpoints
    are saved for consistency with RL experiments.
    """
    def __init__(self, save_freq=10000, log_path="work_dirs", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(self.log_path,
                                     f"checkpoint_{timestamp}_step_{self.n_calls}.zip")
            self.model.save(save_path)
            if self.verbose:
                print(f"[Checkpoint] Step {self.n_calls} saved to {save_path}")
        return True


# ============================================================
#  EntropyDecayCallback
#  ------------------------------------------------------------
#  Gradually decays PPO entropy coefficient over training.
#  (This does not affect IK performance but keeps PPO logs consistent.)
# ============================================================
class EntropyDecayCallback(BaseCallback):
    """
    Decays ent_coef from initial → final over total_steps.
    Used only for logging consistency.
    """
    def __init__(self, ent_coef_initial=0.015, ent_coef_final=0.001,
                 total_steps=500_000, verbose=1):
        super().__init__(verbose)
        self.ent_coef_initial = ent_coef_initial
        self.ent_coef_final = ent_coef_final
        self.total_steps = total_steps

    def _on_step(self):
        progress = min(self.num_timesteps / self.total_steps, 1.0)
        new_ent = self.ent_coef_initial - progress * (
            self.ent_coef_initial - self.ent_coef_final
        )
        self.model.ent_coef = float(new_ent)
        self.logger.record("train/entropy_coef", self.model.ent_coef)
        return True


# ============================================================
#  Main training logic
# ============================================================
log_root = "./results"
run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
log_dir = os.path.join(log_root, run_name)
os.makedirs(log_dir, exist_ok=True)

# Create IK environment
env = gymenv("arm4addforce1104.xml", render=True, log_path=log_dir)

# PPO policy settings (not used for control, but required by SB3)
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    lstm_hidden_size=128,
    n_lstm_layers=1,
    activation_fn=nn.ReLU,
    log_std_init=-1.5,
)

# Create PPO model (policy outputs ignored by environment)
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    learning_rate=2e-4,
    n_steps=256,
    batch_size=128,
    n_epochs=5,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.015,
    vf_coef=0.15,
    max_grad_norm=0.5,
    target_kl=0.15,
    verbose=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log=log_dir
)

# Callbacks
entropy_decay_callback = EntropyDecayCallback()
train_logger_callback = TrainLoggerCallback(log_path=log_dir)

energy_plot_callback = EpisodeEnergyPlotCallback(
    update_freq=1,
    save_interval=10,
    save_path=os.path.join(log_dir, "episode_energy.csv"),
    verbose=1
)

callback_list = CallbackList([
    train_logger_callback,
    energy_plot_callback,
    entropy_decay_callback,
])

# Start training (policy outputs ignored by environment)
model.learn(
    total_timesteps=800_000,
    callback=callback_list,
    tb_log_name="PPO_train"
)

model.save(os.path.join(log_dir, "ppo_model.zip"))
env.save_final_log()
print(f"Training finished. Logs saved to: {log_dir}")
