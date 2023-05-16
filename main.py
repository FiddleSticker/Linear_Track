import sys
import os
import time
import datetime

import constants as c
from Agents.rnn_experiment import RNNExperiment
from Agents.ffn_experiment import FFNExperiment

if __name__ == "__main__":
    network = None
    runs = 0
    size = 0
    path_prefix = ""

    for arg in sys.argv[1:]:
        if arg == "-f":
            network = "ffn"
        elif arg == "-r":
            network = "rnn"
        elif (size_arg := "--size=") in arg:
            size = int(arg.replace(size_arg, ""))
        elif (runs_arg := "--runs=") in arg:
            runs = int(arg.replace(runs_arg, ""))
        elif (path_prefix_arg := "--path-prefix=") in arg:
            path_prefix = str(arg.replace(path_prefix_arg, "")) + "_"
        else:
            raise TypeError(f"UNRECOGNIZED ARGUMENT: {arg}")

    if network == "ffn":
        demo_scene = os.path.join(c.PATH_DATA, f"linear_track_1x{size}")
        exp = FFNExperiment(demo_scene, size, visual_output=False)
    elif network == "rnn":
        demo_scene = os.path.join(c.PATH_DATA, f"linear_track_1x{size}")
        exp = RNNExperiment(demo_scene, size, visual_output=False)
    else:
        raise TypeError("No network type given")

    start_time = time.time()
    print(f"--- start: {datetime.datetime.now()} ---")
    print(exp.run(runs))  # Runs experiment and prints result
    # print(exp.save(os.path.join(c.PATH_DATA, f"{path_prefix}{network}_linear_track_1x{size}.pkl")))
    print(f"--- {time.time() - start_time} seconds ---")

