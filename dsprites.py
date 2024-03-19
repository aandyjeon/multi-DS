import os
import numpy as np
import random

filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

dataset_zip = np.load(filename)

print('Keys in the dataset:', dataset_zip.keys())
imgs         = dataset_zip['imgs']
index        = [idx for idx in range(len(imgs))]
random_index = random.sample(index, len(imgs))
imgs         = imgs[random_index] 
train_imgs   = imgs[:int(0.8 * len(imgs))] 
test_imgs    = imgs[int(0.8 * len(imgs)):] 
latents_values  = dataset_zip['latents_values'][random_index]
train_latents_values = latents_values[:int(0.8 * len(imgs))] 
test_latents_values  = latents_values[int(0.8 * len(imgs)):] 
latents_classes = dataset_zip['latents_classes']

print(f"imgs: {imgs.shape}")
print(f"latents_classes: {latents_classes.shape}")
print(f"latents_values: {latents_values.shape}")

"""
1. Shape: (square, ellipse, heart)
   1, 2, 3
2. Object Color: (red, yellow, blue)
   (255, 0, 0), (255, 255, 0), (0, 0, 255)
3. Background Color: (orange, green, purple)
   (255, 153, 51), (0, 153, 0), (102, 0, 255)
4. Scale: (small: [0.5, 0.6], middle: [0.7, 0.8], big: [0.9, 1])
   (0.5, 0.6), (0.7, 0.8), (0.9, 1)
"""

iid_test   = np.array([])
label_test = []

for shape in ['square', 'ellipse', 'heart']:
    if shape == 'square':  
        _shape = 1 
    elif shape == 'ellipse':
        _shape = 2
    else: 
        _shape = 3

    for scale in ['small', 'middle', 'big']:
        if scale == 'small':
            _scale = 0.5
        elif scale == 'middle':
            _scale = 0.7
        else:
            _scale = 0.9

        index = [idx for idx in range(len(train_imgs)) if (train_latents_values[idx][1] == _shape) and (train_latents_values[idx][2] == _scale)]
        index_sample = random.sample(index, 5000)
        img = train_imgs[index_sample]
        index = [idx for idx in range(len(test_imgs)) if (test_latents_values[idx][1] == _shape) and (test_latents_values[idx][2] == _scale)]
        index_sample = random.sample(index, 10)
        test_img = test_imgs[index_sample]

        for obj_color in ['red', 'yellow', 'blue']:
            if obj_color == 'red':
                _obj_color = [255/255., 0, 0]
            elif obj_color == 'yellow':
                _obj_color = [255/255., 255/255., 0]
            elif obj_color == 'blue':
                _obj_color = [0, 0, 255/255.]

            for bg_color in ['orange', 'green', 'purple']:
                if bg_color == 'orange':
                    _bg_color = [255/255., 153/255., 51/255.]
                elif bg_color == 'green':
                    _bg_color = [0, 153/255., 0]
                elif bg_color == 'purple':
                    _bg_color = [102/255., 0, 255/255.]
        
                output_array = np.zeros((*img.shape, 3))
                mask_1 = img == 1
                mask_0 = img == 0
                output_array[mask_1] = _obj_color
                output_array[mask_0] = _bg_color
        
                output_array_test = np.zeros((*test_img.shape, 3))
                mask_1 = test_img == 1
                mask_0 = test_img == 0
                output_array_test[mask_1] = _obj_color
                output_array_test[mask_0] = _bg_color
  
                print(f"[DEBUG] {output_array.shape}")          
                np.save(f'./dsprites_split/{shape}_{obj_color}_{bg_color}_{scale}.npy', output_array)
                if len(iid_test) == 0:
                    iid_test = output_array_test
                else:
                    iid_test = np.append(iid_test, output_array_test, axis = 0)
                tmp = [shape] * 10
                label_test.append(tmp)

label_test = np.reshape(label_test, -1)
print(f"[DEBUG] {iid_test.shape}")
print(f"[DEBUG] {label_test.shape}")
np.save(f'./dsprites_split/iid_test.npy', iid_test)
np.save(f'./dsprites_split/label_test.npy', label_test)

