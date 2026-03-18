from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from reward_plot_callback import RewardPlotCallback
import pandas as pd
import json
import os
from gym_twofluid_1104 import gymenv
from torch import nn
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Real-time plotting of energy consumption at the end of each episode
class EpisodeEnergyPlotCallback(BaseCallback):
    """
    Records info['epi_energy'] at the end of each episode,
    plots it in real-time, and automatically saves it as CSV every certain number of training steps.
    """
    def __init__(self, update_freq=1, save_interval=5000, save_path="episode_energy.csv", verbose=0):
        super().__init__(verbose)
        self.update_freq = update_freq           # Plot update frequency (in number of episodes)
        self.save_interval = save_interval       # Save frequency (in number of training steps)
        self.save_path = save_path
        self.episode_energies = []
        self.episode_indices = []
        self.last_save_step = 0                  # Last saved training step

        # Real-time plot initialization
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=1.8, label="Episodic Energy")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Energy (epi_energy)")
        self.ax.set_title("Real-time Episode Energy")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.legend()
        plt.show(block=False)

    def _on_step(self):
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        step = self.num_timesteps  # Current global training step

        # Capture energy consumption at the end of each episode
        if dones is not None:
            for done, info in zip(dones, infos):
                if done and "epi_energy" in info:
                    epi_energy = float(info["epi_energy"])
                    self.episode_energies.append(epi_energy)
                    self.episode_indices.append(len(self.episode_energies))
                    if len(self.episode_energies) % self.update_freq == 0:
                        self._update_plot()

        # Save to CSV every save_interval training steps
        if step - self.last_save_step >= self.save_interval:
            self._save_to_csv()
            self.last_save_step = step

        return True

    def _update_plot(self):
        self.line.set_xdata(self.episode_indices)
        self.line.set_ydata(self.episode_energies)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

    def _save_to_csv(self):
        df = pd.DataFrame({
            "Episode": self.episode_indices,
            "Epi_Energy": self.episode_energies
        })
        df.to_csv(self.save_path, index=False)
        if self.verbose:
            print(f"Episode energy table saved ({len(self.episode_energies)} records) → {self.save_path}")

    def _on_training_end(self):
        # Ensure it saves once at the end of training
        self._save_to_csv()
        print(f"Final episode energy saved to {self.save_path}")

# Original training log and model saving Callback 
class TrainLoggerCallback(BaseCallback):
    def __init__(self, save_freq=10000, log_path="work_dirs", verbose=1):
        super(TrainLoggerCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        self.log_file = os.path.join(self.log_path, "progress_log.json")
        self.records = []

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            info = {"step": self.n_calls}
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(self.log_path, f"checkpoint_{timestamp}_step_{self.n_calls}.zip")
            self.model.save(save_path)
            if self.verbose:
                print(f"Model saved at step {self.n_calls} to {save_path}")
        return True

# Entropy Decay Callback
class EntropyDecayCallback(BaseCallback):
    def __init__(self, ent_coef_initial=0.015, ent_coef_final=0.001, total_steps=500_000, verbose=1):
        super().__init__(verbose)
        self.ent_coef_initial = ent_coef_initial
        self.ent_coef_final = ent_coef_final
        self.total_steps = total_steps

    def _on_step(self):
        progress = min(self.num_timesteps / self.total_steps, 1.0)
        new_ent_coef = self.ent_coef_initial - progress * (self.ent_coef_initial - self.ent_coef_final)
        self.model.ent_coef = float(new_ent_coef)
        self.logger.record("train/entropy_coef", self.model.ent_coef)
        if self.verbose and self.num_timesteps % 10000 == 0:
            print(f"Step {self.num_timesteps}: Current entropy coefficient = {self.model.ent_coef:.6f}")
        return True

# Main program part
log_root = "./results"
run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
log_dir = os.path.join(log_root, run_name)
os.makedirs(log_dir, exist_ok=True)

env = gymenv("arm4addforce1104.xml", render=True, log_path=log_dir)

policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256, 128],
        vf=[256, 256, 128]
    ),
    lstm_hidden_size=128,
    n_lstm_layers=1,
    activation_fn=nn.ReLU,
    log_std_init=-1.5
)
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

# Load existing model
model = RecurrentPPO.load("checkpoint_2025-11-17_16-03-30_step_30000.zip", env=env, tensorboard_log=log_dir)

# Define all callbacks
entropy_decay_callback = EntropyDecayCallback(
    ent_coef_initial=0.025,
    ent_coef_final=0.0001,
    total_steps=150_000,
    verbose=1
)
train_logger_callback = TrainLoggerCallback(save_freq=10000, log_path=log_dir, verbose=1)

reward_plot_callback = RewardPlotCallback(update_freq=10)
energy_plot_callback = EpisodeEnergyPlotCallback(
    update_freq=1,
    save_interval=5000,  # Save table every 5000 training steps
    save_path=os.path.join(log_dir, "episode_energy.csv"),
    verbose=1
    )  

callback_list = CallbackList([
    train_logger_callback,
    reward_plot_callback,
    energy_plot_callback,
    entropy_decay_callback,
])

# Start training
model.learn(
    total_timesteps=800_000,
    callback=callback_list,
    tb_log_name="PPO_train"
)
model.save(os.path.join(log_dir, "ppo_model.zip"))
print(f"Model and VecNormalize parameters saved to: {log_dir}")

env.save_final_log()