# Basic imports
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

import constants as c


agents = ["Random", "Feed Forward", "LSTM"]
paths = ["ffn_rand_{}.pkl", "ffn_{}.pkl", "rnn_{}.pkl"]
labels = "4,6,8".split(",")

# Reading Data
data = {}
for agent, path in zip(agents, paths):
    data[agent] = {"y": [], "y_err": []}
    for i in labels:
        try:
            data_path = os.path.join(c.PATH_DATA, path.format(i))
            agent_data = pickle.load(open(data_path, 'rb'))
            agent_data: dict
        except Exception as e:
            print("ERROR LOADING DATA")
            print(e)

        data[agent]["y"].append(agent_data["reward_mse_norm_avg"])
        data[agent]["y_err"].append(np.sqrt(agent_data["reward_var_avg"]) / np.sqrt(10))

# Plotting Data
x = np.arange(len(labels))
bar_width = 0.9 / len(agents)

fig, ax = plt.subplots()
for i, agent in enumerate(agents):
    y = data[agent]["y"]
    y_err = data[agent]["y_err"]
    bar_pos = x - 0.3 + bar_width * i
    bars = ax.bar(bar_pos, y, bar_width, yerr=y_err, label=agent)

ax.set_xlabel("Track Length")
ax.set_ylabel('Trial Reward')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
ax.set_title('Performance by agent and track length')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc=2, prop={"size": 7})

fig.tight_layout()
fig.savefig('performances_4_8.png', dpi=fig.dpi)

# plt.show()
