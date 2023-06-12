# Basic imports
import os
import numpy as np
import pyqtgraph as qg
import tensorflow as tf

from tensorflow.keras import backend as K
# from tensorflow.compat.v1.experimental import output_all_intermediates  # 2.11
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TimeDistributed, Flatten, LSTM

import constants as c
from .base_experiment import Experiment
from Tools.test_trajectory import generate_trajectory

# framework imports
from cobel.networks.network_tensorflow import SequentialKerasNetwork
from Tools.frontends_blender import ImageInterface
from cobel.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import \
    ManualTopologyGraphNoRotation

from cobel.observations.image_observations import ImageObservationBaseline
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor
from cobel.agents.drqn import SimpleDQN

tf.compat.v1.experimental.output_all_intermediates(True)


class RNNExperiment(Experiment):
    def __init__(self, demo_scene: str, length: int, trials: int = c.TRIALS_DEFAULT, memory: bool = True):
        super().__init__(demo_scene, length, trials=trials, memory=memory)

    def build_model(self, input_shape, output_units):
        """
        This function builds a simple network model.

        Parameters
        ----------
        input_shape :                       The network model's input shape.
        output_units :                      The network model's number of output units.

        Returns
        ----------
        model :                             The built network model.
        """
        units = 64
        model = Sequential()
        model.add(Dense(units=units, input_shape=input_shape, activation='tanh'))
        model.add(Dense(units=units, activation='tanh'))
        model.add(Dense(units=units, activation='tanh'))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(units=units, activation='tanh', return_sequences=True))
        model.add(Dense(units=output_units, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # model.summary()

        return model

    def single_run(self) -> dict:
        """
        This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output),
        or by a direct call (in this case, visual output can be used).
        """
        np.random.seed()

        main_window = None
        if self._visual_output:  # if visual output is required, activate an output window
            main_window = qg.GraphicsLayoutWidget(title='Demo: DRQN')
            main_window.show()

        # a dictionary that contains all employed modules
        modules = {}
        modules['world'] = ImageInterface(self.demo_scene)
        modules['observation'] = ImageObservationBaseline(modules['world'], main_window, self._visual_output)
        modules['observation'].format = 'rgb'
        modules['spatial_representation'] = \
            ManualTopologyGraphNoRotation(modules, {'start_nodes': [0], 'goal_nodes': [self.length-1], 'clique_size': 2})
        modules['spatial_representation'].set_visual_debugging(self._visual_output, main_window)
        modules['rl_interface'] = InterfaceBaseline(modules, self._visual_output, self.reward_callback)

        time_horizon = self.max_steps  # how many steps are given to the network

        # initialize monitors
        reward_monitor = RewardMonitor(self.trials, main_window, self._visual_output, [-self.max_steps, self.reward])
        escape_latency_monitor = EscapeLatencyMonitor(self.trials, self.max_steps, main_window, self._visual_output)

        # prepare custom_callbacks
        custom_callbacks = {'on_trial_end': [reward_monitor.update, escape_latency_monitor.update,
                                             lambda _: self.reset_world()]}

        # build model
        model = SequentialKerasNetwork(
            self.build_model((time_horizon,) + modules['rl_interface'].observation_space.shape, 2))

        # initialize RL agent
        rl_agent = SimpleDQN(modules['rl_interface'], 0.3, model=model, time_horizon=time_horizon,
                             custom_callbacks=custom_callbacks)

        # eventually, allow the OAI class to access the robotic agent class
        modules['rl_interface'].rl_agent = rl_agent

        # and allow the topology class to access the rlAgent
        modules['spatial_representation'].rl_agent = rl_agent

        # let the agent learn, with extremely large number of allowed maximum steps
        rl_agent.train(self.trials, self.max_steps)
        # rl_agent.test(1, self.max_steps)

        # Recording results
        result = {
            "reward_mse": self.calc_mse(reward_monitor.reward_trace, self.target),
            "reward_trace": reward_monitor.reward_trace,
            "reward_trace_av": np.mean(reward_monitor.reward_trace),
            "latency_traces": escape_latency_monitor.latency_trace,
            "latency_traces_av": np.mean(escape_latency_monitor.latency_trace),
            "trajectories": self.trajectories
        }
        result["reward_mse_norm"] = 1 - (result["reward_mse"] / (self.worst_case ** 2))
        result["reward_var"] = self.calc_variance(reward_monitor.reward_trace, result["reward_trace_av"])
        result["reward_error"] = 1 - (np.sqrt(result["reward_mse"]) / self.worst_case)

        # Todo
        # outputs = [layer.output for layer in model.model.layers]
        # functors = [tf.keras.backend.function([model.model.input, tf.keras.backend.learning_phase()], [out])
        #             for out in outputs]
        # outputs = [layer.output for layer in rl_agent.model_online.model.layers]
        # functors = [
        #     tf.keras.backend.function([rl_agent.model_online.model.input, tf.keras.backend.learning_phase()], [out])
        #     for out in outputs]

        # self.model_online.predict_on_batch(padded_state)[0, self.current_step]

        # images = generate_trajectory("linear_track_4/images.npy", [0, 1, 2, 3, 2, 1, 0], 16)
        # layer_outs = [func(images) for func in functors]

        # clear keras session (for performance)
        K.clear_session()

        # stop simulation
        # modules['world'].stopBlender()

        # and also stop visualization
        if self._visual_output:
            main_window.close()

        return result
