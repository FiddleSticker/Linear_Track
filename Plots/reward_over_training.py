# Basic imports
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import constants as c


agents = ["Random", "Feed Forward", "LSTM"]
paths = ["ffn_rand_{}.pkl", "ffn_{}.pkl", "lstm_{}.pkl"]
labels = "4,5,6,7,8".split(",")

# Reading Data
data = {}
for agent, path in zip(agents, paths):
    data[agent] = {"reward_traces": [], "reward_traces_averages": []}
    for i in labels:
        try:
            data_path = os.path.join(c.PATH_DATA, path.format(i))
            agent_data = pickle.load(open(data_path, 'rb'))
            agent_data: dict
        except Exception as e:
            print("ERROR LOADING DATA")
            print(e)

        reward_traces = pd.DataFrame(agent_data[c.REWARD_TRACE])
        reward_traces_norm = 1 - ((agent_data[c.TARGET]-reward_traces).abs() / agent_data[c.MAX_DIFF])
        reward_traces_norm = reward_traces_norm.mean()

        # Calculate average on the fly
        reward_traces_avg_online = []
        for idx, _ in enumerate(reward_traces_norm):
            value = reward_traces_norm[max(0, idx - 20):idx+1].mean()
            reward_traces_avg_online.append(value)

        data[agent]["reward_traces"].append(reward_traces_norm)
        data[agent]["reward_traces_averages"].append(reward_traces_avg_online)


# Plotting Data
fig, axs = plt.subplots(nrows=2, ncols=np.floor(len(labels)/2).astype(int), sharex="all", sharey="all")

for ax, (idx, label) in zip(axs.flatten(), enumerate(labels)):
    for agent in agents:
        online_avg = data[agent]["reward_traces_averages"][idx]
        line = ax.plot(online_avg, label=agent)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlabel(f"L = {label}")
    ax.set_yticks(np.linspace(0, 1, 11))
    [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % 2 != 0]

fig.supxlabel("Trials")
fig.supylabel("Average reward")
handles, labels = axs[-1, -1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, 1), loc="upper center", ncol=len(agents))
fig.tight_layout()
fig.subplots_adjust(top=0.925, bottom=0.125)
fig.savefig('reward_over_trial.png', dpi=fig.dpi)

# plt.show()
