import os

# Paths
PATH_ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(PATH_ROOT, "Data")
PATH_AGENTS = os.path.join(PATH_ROOT, "Agents")
PATH_PLOTS = os.path.join(PATH_ROOT, "Plots")
PATH_TOOLS = os.path.join(PATH_ROOT, "Tools")

# Default Values
RUNS_DEFAULT = 1
TRIALS_DEFAULT = 500
MEMORY_DEFAULT = True

# Results keys

MAX_DIFF = "worst_case"
REWARD_TRACE = "reward_trace"
TARGET = "target"
TRAJECTORIES = "trajectories"

# join data
DONT_JOIN = [MAX_DIFF, TARGET]
NO_AVERAGE = DONT_JOIN + [TRAJECTORIES]
