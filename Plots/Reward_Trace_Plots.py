import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

import constants as c


agents = ["rand_f_s", "ffn_s", "rnn"]
labels = "4,6,8".split(",")

# Reading Data
data = {}
for agent in agents:
    data[agent] = {"y": [], "y_err": []}
    for i in labels:
        try:
            path = os.path.join(c.PATH_DATA, f"{agent}_linear_track_1x{i}.pkl")
            agent_data = pickle.load(open(path, 'rb'))
            agent_data: dict
        except Exception as e:
            print("ERROR LOADING DATA")
            print(e)

        data[agent]["y"].append(agent_data["reward_mse_norm_avg"])
        data[agent]["y_err"].append(agent_data["reward_var_avg"] / np.sqrt(100*10))

# Plotting Data
x = np.arange(len(labels))
bar_width = 0.9 / len(agents)

fig, ax = plt.subplots()
for i, agent in enumerate(agents):
    y = data[agent]["y"]
    y_err = data[agent]["y_err"]
    bar_pos = x - 0.3 + bar_width * i
    rnn_bars = ax.bar(bar_pos, y, bar_width, yerr=y_err, label=agent)
# fnn_bars = ax.bar(x - bar_width/2, ffn_mse_norm_avg, bar_width, yerr=ffn_sem, label="Feed Forward")

ax.set_xlabel("Track Length")
ax.set_ylabel('Trail Reward')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
ax.set_title('Performance by agent and track length')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.legend()

fig.tight_layout()
fig.savefig('HDKJSKkjkjhjkLA.png', dpi=fig.dpi)

# plt.show()
