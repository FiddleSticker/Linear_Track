failed = False


def fail(level: str, exception):
    global failed
    failed = True
    print(f"ERROR {level} MODULES")
    print(exception)


def success(level: str):
    print(f"SUCCESSFULLY IMPORTED {level} MODULES")


# Built-in imports
try:
    import time
    import os
    import sys
    import pickle
    from abc import ABC
except ImportError as e:
    fail("BUILT-IN", e)
else:
    success("BUILT-IN")

# Requirement imports
try:
    import numpy as np
    import tensorflow as tf

    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, TimeDistributed, Flatten, LSTM
except ImportError as e:
    fail("REQUIRED", e)
else:
    success("REQUIRED")

# Framework imports
try:
    from cobel.networks.network_tensorflow import SequentialKerasNetwork
    from cobel.frontends.frontends_blender import FrontendBlenderInterface, ImageInterface
    from cobel.frontends.frontends_godot import FrontendGodotTopology
    from cobel.spatial_representations.topology_graphs.simple_topology_graph import GridGraph
    from cobel.observations.image_observations import ImageObservationBaseline
    from cobel.interfaces.baseline import InterfaceBaseline
    from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor
    from cobel.agents.drqn import SimpleDQN
    from cobel.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import \
        ManualTopologyGraphNoRotation
except ImportError as e:
    fail("FRAMEWORK", e)
else:
    success("FRAMEWORK")

if not failed:
    success("ALL")
