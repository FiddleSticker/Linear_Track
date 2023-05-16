import pickle
import matplotlib.pyplot as plt
import numpy as np


RNN_mse = [964.6599999999999, 620.436]
RNN_vars = [0.2]
FNN_mse = [981.418, 670.5759999999999]
FNN_vars = [0.2]

data = {}
try:
    data["rnn"] = pickle.load(
        open("C:/Users/wafor/Desktop/Uni Neu/BACHELORARBEIT/Data/rnn_linear_track_1x6.pkl", 'rb'))
    # data["fnn"] = pickle.load(
    #     open("C:/Users/wafor/Desktop/Uni Neu/BACHELORARBEIT/Data/fnn_linear_track_1x6.pkl", 'rb'))
except Exception as e:
    print("ERROR LOADING DATA")
    print(e)

# rnn_mse_norm_avg = data["rnn"]["reward_mse_norm_avg"]
rnn_mse_norm_avg = data["rnn"]["reward_mse_norm_avg"]
fnn_mse_norm_avg = data["rnn"]["reward_trace_av"]
# fnn_mse_norm_avg = data["fnn"]["reward_mse_norm_avg"]

labels = ["4"]
x = np.arange(1)
bar_width = 0.4

fig, ax = plt.subplots()
rnn_bars = ax.bar(x + bar_width/2, rnn_mse_norm_avg, bar_width, yerr=RNN_vars, label="Recurrent")
fnn_bars = ax.bar(x - bar_width/2, fnn_mse_norm_avg, bar_width, yerr=FNN_vars, label="Feed Forward")

ax.set_xlabel("Track Length")
ax.set_ylabel('Mse')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
ax.set_title('Performance by agent and network type')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
fig.savefig('temp_6_neu.png', dpi=fig.dpi)

plt.show()
