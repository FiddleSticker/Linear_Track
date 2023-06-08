# Imports
import os
import numpy as np
import constants as c

# Parameters
trajectory = [0, 1, 2, 3, 2, 1, 0]
time_horizon = 16
viewport_shape = (1, 30, 3)

# Loading images
images = np.load(os.path.join(c.PATH_DATA, "linear_track_1x4", "images.npy"))

# Creating trajectory
padded_states = np.zeros((1, time_horizon) + viewport_shape)
for idx, image_idx in enumerate(trajectory):
    inv_idx = len(trajectory) - (idx + 1)
    padded_states[0, inv_idx] = images[image_idx]

# Saving
np.save(os.path.join(c.PATH_DATA, "linear_track_1x4", "ideal_trajectory.npy"), padded_states)
