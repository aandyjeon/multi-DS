import os
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

# Constants and look-up dictionaries
START_AZIMUTH = {
(0, 4): 0,(0, 6): 0,(0, 7): 6,(0, 8): 0,(0, 9): 0,(1, 4): 0,(1, 6): 2,(1, 7): 34,(1, 8): 0,
(1, 9): 0,(2, 4): 10,(2, 6): 12,(2, 7): 32,(2, 8): 12,(2, 9): 10,(3, 4): 4,(3, 6): 4,
(3, 7): 4,(3, 8): 2,(3, 9): 2,(4, 4): 4,(4, 6): 6,(4, 7): 6,(4, 8): 6,(4, 9): 6,(0, 0): 0,
(0, 1): 0,(0, 2): 2,(0, 3): 0,(0, 5): 0,(1, 0): 0,(1, 1): 0,(1, 2): 4,(1, 3): 0,(1, 5): 0,
(2, 0): 0,(2, 1): 16,(2, 2): 6,(2, 3): 10,(2, 5): 12,(3, 0): 6,(3, 1): 0,(3, 2): 4,(3, 3): 4,
(3, 5): 4,(4, 0): 4,(4, 1): 4,(4, 2): 4,(4, 3): 4,(4, 5): 6
}


# Load datasets
train = pd.read_parquet('train-00000-of-00001-ba54590c34eb8af1.parquet')
test = pd.read_parquet('test-00000-of-00001-b4af1727fb5b132e.parquet')

# Extract image data and metadata
train_imgs = train['image_lt']
train_latents_values = train[['category', 'instance', 'elevation', 'azimuth', 'lighting']]

test_imgs = test['image_lt']
test_latents_values = test[['category', 'instance', 'elevation', 'azimuth', 'lighting']]

# Convert binary image data to numpy arrays
def convert_images(image_data_series):
    return [np.asarray(Image.open(BytesIO(binary_data['bytes']))) / 255.0 for binary_data in image_data_series]

train_imgs = convert_images(train_imgs)
test_imgs = convert_images(test_imgs)

# Process and save images based on conditions
iid_test = np.array([])
label_test = []

for category, _category in zip(['animal', 'human', 'airplane', 'truck', 'car'], range(5)):
    for instance_train, instance_test in zip([4, 6, 7, 8, 9], [0, 1, 2, 3, 5]):
        s_train = START_AZIMUTH[(_category, instance_train)]
        s_test = START_AZIMUTH[(_category, instance_test)]

        for elevation in [0, 2, 4, 6, 8]:
            for azimuth_train, azimuth_test in zip(range(0, 34, 8), range(0, 34, 8)):
                a_train = (s_train + azimuth_train) % 34
                a_test = (s_test + azimuth_test) % 34

                for lighting in [0, 1, 2, 3, 4]:
                    img = train_imgs[(train_latents_values['category'] == _category) &
                                     (train_latents_values['elevation'] == elevation) &
                                     (train_latents_values['azimuth'] == a_train) &
                                     (train_latents_values['lighting'] == lighting)]
                    _test_img = test_imgs[(test_latents_values['category'] == _category) &
                                          (test_latents_values['elevation'] == elevation) &
                                          (test_latents_values['azimuth'] == a_test) &
                                          (test_latents_values['lighting'] == lighting)]
                    test_img = _test_img.sample(frac=1).head(1)

                    np.save(f'./tmp/{category}_{instance_train}_{elevation}_{azimuth_train}_{lighting}.npy', img)
                    if iid_test.size == 0:
                        iid_test = test_img
                    else:
                        iid_test = np.append(iid_test, test_img, axis=0)
                    label_test.append([category])

label_test = np.reshape(label_test, -1)
print(f"Test images: {len(iid_test)}")
print(f"Test labels: {len(label_test)}")

np.save('./smallnorb_split/iid_test.npy', iid_test)
np.save('./smallnorb_split/label_test.npy', label_test)
