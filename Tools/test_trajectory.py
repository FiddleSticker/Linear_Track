# Imports
import os
import numpy as np
import constants as c


def generate_trajectory(image_src: str, trajectory: list, time_horizon: int):
    viewport_shape = (1, 30, 3)
    images = np.load(os.path.join(c.PATH_DATA, image_src))

    padded_images = np.zeros((1, time_horizon) + viewport_shape)
    trajectory_images = images[trajectory]
    padded_images[0, :trajectory_images.shape[0]] = trajectory_images

    return padded_images


# def generate_trajectories(image_source: str, trajectory: list, time_horizon: int):
#     viewport_shape = (1, 30, 3)
#     trajectory_images = []
#
#     # Loading images
#     images = np.load(os.path.join(c.PATH_DATA, "linear_track_4", "images.npy"))
#
#     for idx_step, step in enumerate(trajectory):
#
#         # Creating trajectory until idx_step
#         padded_states = np.zeros((1, time_horizon) + viewport_shape)
#         trajectory_until_step = trajectory[:idx_step + 1]
#
#         images_until_step = images[trajectory_until_step]
#         padded_states[0, :images_until_step.shape[0]] = images_until_step
#         # for idx, image_idx in enumerate(trajectory_until_step):
#         #     # INV IDX PROB WRONG INDEXES ARE APPENDED (0,1,2,3,nan,nan...)
#         #     inv_idx = len(trajectory_until_step) - (idx + 1)
#         #     padded_states[0, inv_idx] = images[image_idx]
#
#         trajectory_images.append(np.copy(padded_states))
#
#     # Saving
#     np.save(os.path.join(c.PATH_DATA, image_source), np.array(trajectory_images))


if __name__ == "__main__":
    print(generate_trajectory("linear_track_4/images.npy", [0, 1, 2, 3, 2, 1, 0], 16))
