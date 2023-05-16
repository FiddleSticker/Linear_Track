# basic imports
import numpy as np
import pyqtgraph as qg
# tensorflow
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
# from tensorflow.compat.v1.experimental import output_all_intermediates # 2.11
# output_all_intermediates(True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TimeDistributed, Flatten, LSTM
# framework imports
from cobel.networks.network_tensorflow import SequentialKerasNetwork
from cobel.frontends.frontends_godot import FrontendGodotTopology
from cobel.spatial_representations.topology_graphs.simple_topology_graph import GridGraph
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.interfaces.baseline import InterfaceBaseline
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor
from cobel.agents.drqn import SimpleDQN
# local imports

visual_output = True


def build_model(input_shape, number_of_actions):
    '''
    This function builds a simple network model.

    Parameters
    ----------
    input_shape :                       The network model's input shape.
    number_of_actions :                 The network model's number of output units.

    Returns
    ----------
    model :                             The built network model.
    '''
    model = Sequential()
    model.add(Dense(units=64, input_shape=input_shape, activation='tanh'))
    model.add(Dense(units=64, activation='tanh'))
    model.add(Dense(units=64, activation='tanh'))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=64, activation='tanh', return_sequences=True))
    model.add(Dense(units=number_of_actions, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model

def reward_callback(values):
    '''
    This is a callback function that defines the reward provided to the robotic agent.
    Note: this function has to be adopted to the current experimental design.

    Parameters
    ----------
    values :                            A dict of values that are transferred from the OAI module to the reward function. This is flexible enough to accommodate for different experimental setups.

    Returns
    ----------
    reward :                            The reward that will be provided.
    end_trial :                         Flag indicating whether the trial ended.
    '''
    # the standard reward for each step taken is negative, making the agent seek short routes
    reward = -1.0
    if values['current_node'].goal_node:
        reward = 10.0

    reward = .0
    if values['current_node'].goal_node:
        reward = 1.0

    return reward, values['current_node'].goal_node

def single_run():
    '''
    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: Godot with Grid Graph')

    # set godot path
    godot_path = 'C:/Users/wafor/Desktop/Godot/godot_simulator_build/godot.exe'

    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendGodotTopology('room.tscn', godot_path)
    modules['world'].min_x, modules['world'].max_x = -0.5, 0.5
    modules['world'].min_y, modules['world'].max_y = -0.5, 0.5
    modules['observation'] = ImageObservationBaseline(modules['world'], main_window, visual_output)
    modules['observation'].format = 'rgb'
    modules['spatial_representation'] = GridGraph(n_nodes_x=4, n_nodes_y=4, start_nodes=[0], goal_nodes=[15],
                                                  visual_output=True, world_module=modules['world'],
                                                  use_world_limits=True, observation_module=modules['observation'])
    modules['spatial_representation'].set_visual_debugging(main_window)
    modules['rl_interface'] = InterfaceBaseline(modules, visual_output, reward_callback)

    # amount of trials
    number_of_trials = 100
    # maximum steos per trial
    max_steps = 30

    # initialize reward monitor
    reward_monitor = RewardMonitor(number_of_trials * 2, main_window, visual_output, [0, 1])
    el_monitor = EscapeLatencyMonitor(number_of_trials * 2, max_steps, main_window, visual_output)

    # prepare custom_callbacks
    custom_callbacks = {'on_trial_end': [reward_monitor.update, el_monitor.update]}

    # build model
    model = SequentialKerasNetwork(build_model((max_steps,)+modules['rl_interface'].observation_space.shape, 4))

    # initialize RL agent
    rl_agent = SimpleDQN(modules['rl_interface'], epsilon=0.3, beta=5, gamma=0.99, model=model, time_horizon=max_steps, custom_callbacks=custom_callbacks)

    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent

    # and allow the topology class to access the rlAgent
    modules['spatial_representation'].rl_agent = rl_agent

    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(number_of_trials , max_steps)

    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.test(number_of_trials, max_steps)

    # clear keras session (for performance)
    K.clear_session()

    # stop simulation
    modules['world'].stop_godot()

    # and also stop visualization
    if visual_output:
        main_window.close()

    # clear Keras session (for performance)
    K.clear_session()

if __name__ == '__main__':
    single_run()
