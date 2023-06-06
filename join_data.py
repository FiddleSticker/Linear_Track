import os
import pickle
import numpy as np

import constants as c

paths = [os.path.join(c.PATH_DATA, f"rnn_8_{i}.pkl") for i in [1, 2, 3, 4]]
new_dict_path = os.path.join(c.PATH_DATA, f"rnn_8.pkl")
new_dict = {}

for path in paths:
    try:
        agent_data = pickle.load(open(path, 'rb'))
        agent_data: dict
    except FileNotFoundError as e:
        print(e)
        break

    for key, item in agent_data.items():
        if "avg" not in key:
            if not isinstance(item, list):
                new_dict[key] = item
            else:
                if key not in new_dict:
                    new_dict[key] = item
                else:
                    new_dict[key].extend(item)

# re-calculating averages
new_dict_avg = {}
for key, item in new_dict.items():
    if key not in c.NO_AVERAGE:
        new_dict_avg[f"{key}_avg"] = np.mean(item)
new_dict.update(new_dict_avg)

try:
    with open(new_dict_path, "wb") as file:
        pickle.dump(new_dict, file)
        print(new_dict)
        print(f"JOINED DATA AT: {new_dict_path}")
except Exception as e:
    print(e)

