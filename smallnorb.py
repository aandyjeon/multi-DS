import os
import numpy as np
import random
import pandas as pd

from PIL import Image
from io import BytesIO

"""
1. 5 generic categories: four-legged animals, human figures, airplanes, trucks, and cars. 
   [0, 1, 2, 3, 4]
2. 5 instances
   train: [4, 6, 7, 8, 9]
   test : [0, 1, 2, 3, 5]
3. 9 elevations (30 to 70 degrees every 5 degrees)
   [0, 1, 2, 3, 4, 5, 6, 7, 8]
4. 18 azimuths (0 to 340 every 20 degrees). 
   [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
5. 6 lighting conditions
   [0, 1, 2, 3, 4, 5]

The training set is composed of 5 instances of each category (instances 4, 6, 7, 8 and 9), 
and the test set of the remaining 5 instances (instances 0, 1, 2, 3, and 5).
"""

START_AZIMUTH = {(0, 4): 0,(0, 6): 0,(0, 7): 6,(0, 8): 0,(0, 9): 0,(1, 4): 8,(1, 6): 2,(1, 7): 6,
(1, 8): 2,(1, 9): 8,(2, 4): 10,(2, 6): 12,(2, 7): 32,(2, 8): 12,(2, 9): 10,(3, 4): 4,
(3, 6): 4,(3, 7): 4,(3, 8): 2,(3, 9): 2,(4, 4): 4,(4, 6): 6,(4, 7): 6,(4, 8): 6,
(4, 9): 6,(0, 0): 0,(0, 1): 0,(0, 2): 2,(0, 3): 0,(0, 5): 0,(1, 0): 8,(1, 1): 0,
(1, 2): 4,(1, 3): 0,(1, 5): 8,(2, 0): 0,(2, 1): 16,(2, 2): 6,(2, 3): 10,(2, 5): 12,
(3, 0): 6,(3, 1): 0,(3, 2): 4,(3, 3): 4,(3, 5): 4,(4, 0): 4,(4, 1): 4,(4, 2): 4,
(4, 3): 4,(4, 5): 6}

train = pd.read_parquet('train-00000-of-00001-ba54590c34eb8af1.parquet')
test  = pd.read_parquet('test-00000-of-00001-b4af1727fb5b132e.parquet')

# TRAIN
train_imgs = train['image_lt']
train_latents_values = train[['category', 'instance', 'elevation', 'azimuth','lighting']]

# TEST
test_imgs = test['image_lt']
test_latents_values = test[['category', 'instance', 'elevation', 'azimuth','lighting']]

for i, binary_data in enumerate(train_imgs):
    image_file = BytesIO(binary_data['bytes'])
    train_imgs[i] =  np.asarray(Image.open(image_file))/255.

for i, binary_data in enumerate(test_imgs):
    image_file = BytesIO(binary_data['bytes'])
    test_imgs[i] =  np.asarray(Image.open(image_file))/255.

iid_test   = np.array([])
label_test = []
num = 0

for category in ['animal', 'human', 'airplane', 'truck', 'car']:
    if category == 'animal':  
        _category = 0 
    elif category == 'human':
        _category = 1
    elif category == 'airplane': 
        _category = 2
    elif category == 'truck': 
        _category = 3
    elif category == 'car': 
        _category = 4

    for instance_train, instance_test in zip([4, 6, 7, 8, 9], [0, 1, 2, 3, 5]):

        s_train = START_AZIMUTH[(_category, instance_train)]
        s_test  = START_AZIMUTH[(_category, instance_test)]

        for elevation in [0, 2, 4, 6, 8]:
    
            for azimuth_train, azimuth_test in zip([0, 8, 16, 24, 32], [0, 8, 16, 24, 32]):
                a_train = (s_train + azimuth_train) % 34
                a_test = (s_test + azimuth_test) % 34
    
                for lighting in [0, 1, 2, 3, 4]:
            
                    img = train_imgs[(train_latents_values['category'] == _category) & (train_latents_values['elevation'] == elevation) & (train_latents_values['azimuth'] == a_train) & (train_latents_values['lighting'] == lighting) & (train_latents_values['instance'] == instance_train)]
                    _test_img = test_imgs[(test_latents_values['category'] == _category) & (test_latents_values['elevation'] == elevation) & (test_latents_values['azimuth'] == a_test) & (test_latents_values['lighting'] == lighting) & (test_latents_values['instance'] == instance_test)]
                    test_img = _test_img.sample(frac = 1)
                    test_img = test_img.head(1)
      
                    np.save(f'./tmp/{category}_{instance_train}_{elevation}_{azimuth_train}_{lighting}.npy', img)
                    if len(iid_test) == 0:
                        iid_test = test_img
                    else:
                        iid_test = np.append(iid_test, test_img, axis = 0)
                    tmp = [category] * 1
                    label_test.append(tmp)

label_test = np.reshape(label_test, -1)
print(f"[DEBUG] test img: {len(iid_test)}")
print(f"[DEBUG] test label: {len(label_test)}")
np.save(f'./smallnorb_split/iid_test.npy', iid_test)
np.save(f'./smallnorb_split/label_test.npy', label_test)
