import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2

# Load the dataset annotations
df = pd.read_csv('./CelebAMask-HQ-attribute-anno.txt', delim_whitespace=True, header=0)

path = './CelebA-HQ-img'
img_files = os.listdir(path)

# Define a function to load and process a single image
def load_and_process_image(img_file):
    img_path = os.path.join(path, img_file)
    img = cv2.cvtColor(cv2.resize(cv2.imread(img_path), dsize=(256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    return {os.path.splitext(img_file)[0]: np.array(img)}

# Use ThreadPoolExecutor to load and process images concurrently
with ThreadPoolExecutor() as executor:
    _imgs = list(tqdm(executor.map(load_and_process_image, img_files), total=len(img_files)))

# Combine the dictionaries into one for easier access
combined_dict = {k: v for d in _imgs for k, v in d.items()}

# Prepare subsets based on combinations of attributes
iid_test = []
label_test = []

attributes_combinations = [(male, black_hair, smiling, straight_hair)
                           for male in [-1, 1]
                           for black_hair in [-1, 1]
                           for smiling in [-1, 1]
                           for straight_hair in [-1, 1]]

for attrs in tqdm(attributes_combinations):
    male, black_hair, smiling, straight_hair = attrs
    filtered_df = df.loc[(df['Male'] == male) & 
                         (df['Black_Hair'] == black_hair) & 
                         (df['Smiling'] == smiling) & 
                         (df['Straight_Hair'] == straight_hair)]
    
    if len(filtered_df) < 300:
        continue  # Skip combinations with fewer than 300 samples

    indices = random.sample(list(filtered_df.index), 300)
    imgs = np.array([combined_dict[str(index)] for index in indices if str(index) in combined_dict])

    # Split into training and testing
    img, test_img = imgs[:-50]/255., imgs[-50:]/255.

    # Save processed images
    np.save(f'./celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy', img)
    
    # Update test set and labels
    iid_test.append(test_img)
    label_test.extend([male] * len(test_img))

# Convert lists to numpy arrays
iid_test = np.concatenate(iid_test, axis=0)
label_test = np.array(label_test)

# Save test images and labels
np.save('./celeba_split/iid_test.npy', iid_test)
np.save('./celeba_split/label_test.npy', label_test)

print(f"[DEBUG] iid_test shape: {iid_test.shape}")
print(f"[DEBUG] label_test shape: {label_test.shape}")
