import os
import numpy as np
import pandas as pd
import random
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import cv2

df = pd.read_csv('./CelebAMask-HQ-attribute-anno.txt', delim_whitespace=True, header=0)

#df.index = df.index.map(lambda x: int(x.split('.')[0]))
#print(df.index)

path = './CelebA-HQ-img'
img_files = os.listdir(path)

def load_and_process_image(img_file):
    img_path = os.path.join(path, img_file)
    img = cv2.cvtColor(cv2.resize(cv2.imread(img_path), dsize = (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)

    return {img_path.split('/')[-1]: np.array(img)}

with ThreadPoolExecutor() as executor:
    _imgs = list(executor.map(load_and_process_image, img_files))

combined_dict = {}
for d in _imgs:
    combined_dict.update(d)

iid_test   = np.array([])
label_test = []

for male in [-1, 1]:
    for black_hair in [-1, 1]:
        for smiling in [-1, 1]:
            for straight_hair in [-1, 1]:
                _indices = list(df.loc[(df['Male'] == male) & (df['Black_Hair'] == black_hair) & (df['Smiling'] == smiling) & (df['Straight_Hair'] == straight_hair)].index)
                indices = random.sample(_indices, 300)
                imgs = []
                for index in indices:
                    imgs.append(combined_dict[index])
                imgs = np.array(imgs)
                img, test_img = imgs[:-50]/255., imgs[-50:]/255.
  
                print(f"[DEBUG] {img.shape}")          
                np.save(f'./celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy', img)
                if len(iid_test) == 0:
                    iid_test = test_img
                else:
                    iid_test = np.append(iid_test, test_img, axis = 0)
                tmp = [str(male)] * 50
                label_test.append(tmp)

label_test = np.reshape(label_test, -1)
print(f"[DEBUG] {iid_test.shape}")
print(f"[DEBUG] {label_test.shape}")
np.save(f'./celeba_split/iid_test.npy', iid_test)
np.save(f'./celeba_split/label_test.npy', label_test)
