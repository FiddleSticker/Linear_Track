import sys
import os
import time
import datetime
import argparse

import constants as c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network_type", type=str, choices=["random", "fnn", "rnn", "lstm"], metavar="network_type",
                        help="type of network to run the experiment with. Available networks: random, fnn, rnn, lstm")
    parser.add_argument("length", type=int, help="Length of the linear track")
    parser.add_argument("-r", "--runs", type=int, default=c.RUNS_DEFAULT, metavar="",
                        help=f"number of repetitions of experiment (Default = {c.RUNS_DEFAULT})")
    parser.add_argument("-t", "--trials", type=int, default=c.TRIALS_DEFAULT, metavar="",
                        help=f"number of trials per run. Default = {c.TRIALS_DEFAULT}")
    parser.add_argument("-p", "--path", type=str, metavar="", default=None,
                        help="path to which experiment results are written")
    parser.add_argument("--memory", default=True, action="store_true",
                        help=f"run memory experiment (Default = {c.MEMORY_DEFAULT})")
    parser.add_argument("--no-memory", dest="memory", action="store_false",
                        help="run simple experiment")
    args = parser.parse_args()

    network = args.network_type
    length = args.length
    runs = args.runs
    trials = args.trials
    memory = args.memory
    save_path = args.path

    start_time = time.time()
    print(f"--- start: {datetime.datetime.now()} ---")

    from Agents.rnn_experiment import RNNExperiment
    from Agents.fnn_experiment import FNNExperiment

    if network == "random":
        demo_scene = os.path.join(c.PATH_DATA, f"linear_track_{length}")
        exp = FNNExperiment(demo_scene, length, trials=trials, memory=memory, epsilon=1)
    elif network == "fnn":
        demo_scene = os.path.join(c.PATH_DATA, f"linear_track_{length}")
        exp = FNNExperiment(demo_scene, length, trials=trials, memory=memory)
    elif network in ["rnn", "lstm"]:
        demo_scene = os.path.join(c.PATH_DATA, f"linear_track_{length}")
        exp = RNNExperiment(demo_scene, length, trials=trials, memory=memory)
    else:
        raise TypeError("No network type given")

    print(exp.run(runs))  # Runs experiment and prints result

    if not save_path:
        runs_string = ""
        if runs != c.RUNS_DEFAULT:
            runs_string = f"_r{runs}"

        trials_string = ""
        if trials != c.TRIALS_DEFAULT:
            trials_string = f"_t{trials}"

        memory_string = ""
        if not memory:
            memory_string = f"_no_mem"

        save_path = f"{network}{trials_string}{memory_string}_{length}.pkl"

    print(exp.save(os.path.join(c.PATH_DATA, save_path)))

    print(f"--- {time.time() - start_time} seconds ---")
