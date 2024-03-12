import os
import numpy as np

def load_and_append_data(category, instance, elevation, azimuth, lightening):
    """Load data from a file and append it to an existing array."""
    file_path = f'./tmp/{category}_{instance}_{elevation}_{azimuth}_{lightening}.npy'
    data = np.load(file_path, allow_pickle=True)
    return data

# List of categories to process
categories = ['animal', 'human', 'airplane', 'car', 'truck']

# Initialize dictionaries to store data arrays by category
data_arrays = {category: np.array([]) for category in categories}

# Iterate through each condition
for elevation in [0, 2, 4, 6, 8]:
    for azimuth in [0, 8, 16, 24, 32]:
        for lightening in [0, 1, 2, 3, 4]:
            # Reset arrays for each condition
            for category in categories:
                data_arrays[category] = np.array([])
            
            for instance in [4, 6, 7, 8, 9]:
                for category in categories:
                    if data_arrays[category].size == 0:
                        data_arrays[category] = load_and_append_data(category, instance, elevation, azimuth, lightening)
                    else:
                        new_data = load_and_append_data(category, instance, elevation, azimuth, lightening)
                        data_arrays[category] = np.append(data_arrays[category], new_data, axis=0)

            # Save aggregated data for each category and condition
            for category in categories:
                np.save(f"./smallnorb_split/{category}_{elevation}_{azimuth}_{lightening}.npy", data_arrays[category])

