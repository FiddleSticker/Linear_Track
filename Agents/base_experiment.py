# Basic imports
import time
import os
import pickle
from abc import ABC
import numpy as np

import constants as c


class Experiment(ABC):
    def __init__(self, demo_scene: str, length: int, trials: int = c.TRIALS_DEFAULT, illuminate: bool = False):
        # Experiment Parameters
        self.demo_scene = demo_scene
        self.length = length
        self.trials = trials
        self.reward = 10
        self.penalty = 0.1
        self.max_steps = self.length ** 2  # 30
        self.target = self.reward - self.penalty * (2 * self.length - 3)
        self.worst_case = np.abs(self.target - self.penalty * (-self.max_steps))  # maximum difference to target
        self.max_min = self.penalty * self.max_steps
        self.max_diff = np.abs(self.target - self.penalty * (-self.max_steps))

        self.illuminate = illuminate
        self._visual_output = False
        self.trajectory = []

        self.modules = {}
        self._reached_end = False
        self._cached_results = None

    def single_run(self) -> dict:
        raise NotImplementedError

    def build_model(self, input_shape, output_units):
        raise NotImplementedError

    def reward_callback(self, values):
        """
        This is a callback function that defines the reward provided to the robotic agent.
        Note: this function has to be adopted to the current experimental design.

        | **Args**
        | values:                       A dict of values that are transferred from the OAI module to the reward function.
        This is flexible enough to accommodate for different experimental setups.
        """
        reward = -self.penalty  # the standard reward for each step taken is negative, making the agent seek short routes
        end_trial = False

        self.trajectory.append(values['current_node'].index)

        if values['current_node'].goal_node:
            self._reached_end = True

        if self._reached_end and values['current_node'].start_node:
            reward = self.reward
            end_trial = True
            self.reset_world(values["modules"]["world"])

        return reward, end_trial

    def reset_world(self, world):
        self._reached_end = False
        self.trajectory = []


    @staticmethod
    def calc_mse(x, y) -> float:
        mse = float(np.mean((y-x)**2))
        return mse

    @staticmethod
    def calc_variance(x, y):
        var = (y-x)**2 / (len(x) - 1)
        return var

    def run(self, runs: int = 1, wait: int = 0) -> dict:
        results = {}
        for i in range(runs):
            print(f"Run {i+1}/{runs}")
            result = self.single_run()

            # recording results from each single run
            for key, item in result.items():
                if key not in results:
                    results[key] = []
                results[key].append(item)

            time.sleep(wait)  # give Blender time to restart

        # calculating averages
        results_avg = {}
        for key, item in results.items():
            results_avg[f"{key}_avg"] = np.mean(item)
        results.update(results_avg)
        results["target"] = self.target
        results["worst_case"] = self.worst_case

        self._cached_results = results
        return results

    def save(self, path: str) -> str:
        if self._cached_results:
            if not path.endswith(".pkl"):
                path += ".pkl"
            try:
                with open(path, "wb") as file:
                    pickle.dump(self._cached_results, file)
                    return path
            except Exception as e:
                print(e)

        return ""
