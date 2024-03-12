import os
import numpy as np
import random
import h5py

# Define function to load data from .h5 file
def load_data_from_h5(filename):
    with h5py.File(filename, "r") as f:
        print("Keys: %s" % f.keys())
        imgs_key, label_key = list(f.keys())
        imgs = f[imgs_key][()]
        latents_values = f[label_key][()]
    return imgs, latents_values

filename = './3dshapes.h5'
imgs, latents_values = load_data_from_h5(filename)

# Define selection criteria based on dataset attributes
shapes = [0, 1, 2, 3]
scales = {
    'tiny': 0.75,
    'small': 0.9642857142857143,
    'middle': 1.0357142857142856,
    'big': 1.1785714285714286
}
obj_colors = [0, 0.1, 0.2, 0.3]
bg_colors = [0, 0.1, 0.2, 0.3]

# Initialize arrays for storing test images and labels
iid_test = []
label_test = []

# Process the dataset
for shape in shapes:
    for scale_name, scale_value in scales.items():
        for obj_color in obj_colors:
            for bg_color in bg_colors:
                # Filter images based on selection criteria
                index = [idx for idx, vals in enumerate(latents_values) if
                         (vals[4] == shape) and (vals[3] == scale_value) and
                         (vals[2] == obj_color) and (vals[1] == bg_color) and
                         (vals[0] == bg_color)]
                if len(index) < 15:
                    continue  # Skip if there are not enough samples
                index_sample = random.sample(index, 15)
                
                # Split into training and testing
                img, test_img = imgs[index_sample][:10] / 255., imgs[index_sample][10:] / 255.

                # Save processed images
                np.save(f'./shapes3d_split/{shape}_{obj_color}_{bg_color}_{scale_name}.npy', img)
                
                # Aggregate test images and labels
                iid_test.append(test_img)
                label_test.extend([shape] * 5)  # Each shape gets a label

# Convert lists to arrays and reshape label_test for compatibility
iid_test = np.concatenate(iid_test, axis=0)
label_test = np.array(label_test).reshape(-1)

# Save test images and labels
np.save('./shapes3d_split/iid_test.npy', iid_test)
np.save('./shapes3d_split/label_test.npy', label_test)

print(f"[DEBUG] iid_test shape: {iid_test.shape}")
print(f"[DEBUG] label_test shape: {label_test.shape}")

