import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardPlotCallback(BaseCallback):
    def __init__(self, update_freq=100, verbose=0):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.reward_items = {
            "reward_total": [],
            "reward_progress": [],
            "reward_accuracy": [],
            "reward_sparse": [],
            "reward_energy": [],
            "reward_smooth": [],
            "reward_lip": [],
            "reward_energy_total" :[],
            "alignment_reward":[],
            "terminal_reward":[],
        }

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def _on_step(self):
        # Extract reward information from info
        info = self.locals["infos"][0]
        for key in self.reward_items:
            if key in info:
                self.reward_items[key].append(info[key])

        # Dynamic plotting update
        if self.n_calls % self.update_freq == 0:
            self.ax.clear()
            for key, data in self.reward_items.items():
                if len(data) > 1:
                    self.ax.plot(data, label=key)

            self.ax.set_title("Reward Components Over Training Steps")
            self.ax.set_xlabel("Environment Steps")
            self.ax.set_ylabel("Reward Value")
            self.ax.legend()
            self.ax.grid(True)
            plt.pause(0.01)

        return True

    def _on_training_end(self):
        plt.ioff()
        plt.show()
