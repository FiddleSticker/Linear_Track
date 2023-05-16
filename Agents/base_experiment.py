# Basic imports
import time
import os
import pickle
from abc import ABC
import numpy as np


class Experiment(ABC):
    def __init__(self, demo_scene: str, length: int, trials: int = 100, illuminate: bool = False,
                 visual_output: bool = True):
        # Experiment Parameters
        self.demo_scene = demo_scene
        self.length = length
        self.trials = trials
        self.target = 10 - 2 * self.length + 3
        # self.target = 10 - self.length + 2
        self.max_steps = self.length ** 2  # 30
        self.worst_case = np.abs(self.target - (-self.max_steps))

        self.illuminate = illuminate
        self._visual_output = visual_output

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
        reward = -1.0  # the standard reward for each step taken is negative, making the agent seek short routes
        end_trial = False

        if values['current_node'].goal_node:
            # reward = 10.0
            # end_trial = True

            self._reached_end = True
            # if self.illuminate:
            #     values["modules"]["world"].setIllumination("Sun", [1, 0, 0])

        if self._reached_end and values['current_node'].start_node:
            reward = 10.0
            end_trial = True
            self.reset_world(values["modules"]["world"])

        return reward, end_trial

    def reset_world(self, world):
        self._reached_end = False
        # if self.illuminate:
        #     world.setIllumination("Sun", 3 * [1])

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
