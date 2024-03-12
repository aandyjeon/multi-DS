import os
import numpy as np
import random

# Load the dataset
filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
dataset_zip = np.load(filename)
print('Keys in the dataset:', dataset_zip.keys())

# Shuffle dataset
def shuffle_dataset(imgs, latents_values):
    index = np.arange(len(imgs))
    np.random.shuffle(index)
    return imgs[index], latents_values[index]

imgs, latents_values = shuffle_dataset(dataset_zip['imgs'], dataset_zip['latents_values'])
train_split = int(0.8 * len(imgs))

# Split dataset into training and testing
train_imgs, test_imgs = imgs[:train_split], imgs[train_split:]
train_latents_values, test_latents_values = latents_values[:train_split], latents_values[train_split:]

print(f"imgs: {imgs.shape}")
print(f"latents_classes: {dataset_zip['latents_classes'].shape}")
print(f"latents_values: {latents_values.shape}")

# Initialize arrays for test set and labels
iid_test = []
label_test = []

# Mapping for shapes and scales
shapes = {'square': 1, 'ellipse': 2, 'heart': 3}
scales = {'small': 0.5, 'middle': 0.7, 'big': 0.9}

# Function to process and save images based on specified attributes
def process_and_save_images(imgs, latents_values, shape_name, scale_name, obj_color, bg_color):
    # Filter images based on shape and scale
    shape, scale = shapes[shape_name], scales[scale_name]
    filtered_indexes = [i for i, lv in enumerate(latents_values) if lv[1] == shape and lv[2] == scale]
    filtered_imgs = imgs[filtered_indexes]

    # Apply colors
    output_array = np.zeros((*filtered_imgs.shape, 3))
    mask_1 = filtered_imgs == 1
    mask_0 = filtered_imgs == 0
    output_array[mask_1] = obj_color
    output_array[mask_0] = bg_color

    return output_array

# Object and background colors
obj_colors = {'red': [1.0, 0, 0], 'yellow': [1.0, 1.0, 0], 'blue': [0, 0, 1.0]}
bg_colors = {'orange': [1.0, 153/255., 51/255.], 'green': [0, 153/255., 0], 'purple': [102/255., 0, 1.0]}

# Iterate over shapes, scales, object colors, and background colors
for shape in shapes:
    for scale in scales:
        for obj_color_name, obj_color in obj_colors.items():
            for bg_color_name, bg_color in bg_colors.items():
                # Process training images
                output_array = process_and_save_images(train_imgs, train_latents_values, shape, scale, obj_color, bg_color)
                np.save(f'./dsprites_split/{shape}_{obj_color_name}_{bg_color_name}_{scale}.npy', output_array)

                # Process testing images
                output_array_test = process_and_save_images(test_imgs, test_latents_values, shape, scale, obj_color, bg_color)
                iid_test.append(output_array_test)
                label_test += [shape] * output_array_test.shape[0]

# Convert lists to numpy arrays
iid_test = np.concatenate(iid_test, axis=0)
label_test = np.array(label_test)

print(f"[DEBUG] iid_test shape: {iid_test.shape}")
print(f"[DEBUG] label_test shape: {label_test.shape}")

# Save test images and labels
np.save('./dsprites_split/iid_test.npy', iid_test)
np.save('./dsprites_split/label_test.npy', label_test)

