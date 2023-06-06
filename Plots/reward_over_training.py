# Basic imports
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import constants as c


agents = ["Random", "Feed Forward", "LSTM"]
paths = ["ffn_rand_{}.pkl", "ffn_{}.pkl", "rnn_{}.pkl"]
labels = "4,6,8".split(",")

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
            value = reward_traces_norm[:idx+1].sum() / (idx+1)
            reward_traces_avg_online.append(value)

        data[agent]["reward_traces"].append(reward_traces_norm)
        data[agent]["reward_traces_averages"].append(reward_traces_avg_online)


# Plotting Data
x = np.arange(len(labels))

fig, axs = plt.subplots(nrows=1, ncols=3, sharex="all", sharey="all")


for ax, (idx, label) in zip(axs, enumerate(labels)):
    for agent in agents:
        online_avg = data[agent]["reward_traces_averages"][idx]
        line = ax.plot(online_avg, label=agent)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)



# fig, axs = plt.subplots(1, 3)
# for agent in agents:
#     for ax, online_avg in zip(axs, data[agent]["reward_traces_averages"]):
#         line = ax.plot(online_avg)
#
#         ax.set_xlabel("Track Length")
#         ax.set_ylabel('Trial Reward')
#         ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
        # ax.set_title('Performance by agent over trials')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# fig.set_xlabel()

fig.supxlabel("Trials")
fig.supylabel("Average reward")
handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.05), loc="lower center", ncol=len(agents))
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.savefig('lolopopo.png', dpi=fig.dpi)

# plt.show()
