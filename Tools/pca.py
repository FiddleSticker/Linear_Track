import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from Tools.test_trajectory import generate_trajectory


def pca(length, time_horizon, functors, trajectories):
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    ax.set_xlabel(f"PC 1")
    ax.set_ylabel(f"PC 2")
    ax.set_zlabel(f"PC 3")
    colors_pre = np.array(
        [(0.0, 0.5, 1.0), (0.0, 0.6, 0.9), (0.0, 0.7, 0.8), (0.0, 0.8, 0.7), (0.0, 0.9, 0.6), (0.0, 1.0, 0.5)])
    colors_post = np.array(
        [(1.0, 0.5, 0.0), (1.0, 0.4, 0.1), (1.0, 0.3, 0.2), (1, 0.2, 0.3), (1.0, 0.1, 0.4), (1.0, 0.0, 0.5)])
    for trajectory in trajectories:
        if len(trajectory) > time_horizon:
            trajectory = trajectory[-time_horizon:]
        images = generate_trajectory(f"linear_track_{length}/images.npy", trajectory, time_horizon) / 255
        layer_outs = [func(images) for func in functors]
        layer_outs = layer_outs[-2][0][0][:len(trajectory)]
        X_reduced = PCA(n_components=3).fit_transform(layer_outs)
        if (length - 1) in trajectory:
            goal_time_index = trajectory.index(length - 1)
        else:
            goal_time_index = len(trajectory)
        colors = list(
            np.append(colors_pre[trajectory[:goal_time_index]], colors_post[trajectory[goal_time_index:]], axis=0))
        ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            X_reduced[:, 2],
            color=colors
        )
    plt.interactive(False)
    plt.show()
