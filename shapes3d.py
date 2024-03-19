import os
import numpy as np
import random
import h5py

filename = './3dshapes.h5'

with h5py.File(filename, "r") as f:
    print("Keys: %s" % f.keys())
    imgs  = list(f.keys())[0]
    label = list(f.keys())[1]
    imgs = f[imgs][()]
    latents_values = f[label][()]

"""
1. floor hue: 10 values linearly spaced in [0, 1]
   0 ~ 0.9
   USE: [0, 0.1, 0.2, 0.30000000000000004]
2. wall hue: 10 values linearly spaced in [0, 1]
   0 ~ 0.9
   USE: [0, 0.1, 0.2, 0.30000000000000004]
3. object hue: 10 values linearly spaced in [0, 1]
   0 ~ 0.9
   USE: [0, 0.1, 0.2, 0.30000000000000004]
4. scale: 8 values linearly spaced in [0, 1]
  0.75, 0.8214285714285714, 0.9642857142857143, 1.0357142857142856, 0.8928571428571428, 1.1071428571428572, 1.1785714285714286, 1.25
  USE: [0.75, 0.9642857142857143, 1.0357142857142856, 1.1785714285714286]
5. shape: 4 values in [0, 1, 2, 3]
"""

iid_test   = np.array([])
label_test = []

for shape in [0, 1, 2, 3]:

    for scale in ['tiny', 'small', 'middle', 'big']:
        if scale == 'tiny':
            _scale = 0.75
        elif scale == 'small':
            _scale = 0.9642857142857143
        elif scale == 'middle':
            _scale = 1.0357142857142856
        else:
            _scale = 1.1785714285714286


        for obj_color in [0, 0.1, 0.2, 0.3]:
            _obj_color = obj_color
            if obj_color == 0.3:
                _obj_color = 0.30000000000000004

            for bg_color in [0, 0.1, 0.2, 0.3]:
                _bg_color = bg_color
                if bg_color == 0.3:
                    _bg_color = 0.30000000000000004
        
                index = [idx for idx in range(len(imgs)) if (latents_values[idx][4] == shape) and (latents_values[idx][3] == _scale) and (latents_values[idx][2] == _obj_color) and (latents_values[idx][1] == _bg_color) and (latents_values[idx][0] == _bg_color)]
                index_sample = random.sample(index, 15)
                img, test_img = imgs[index_sample][:10]/255., imgs[index_sample][10:]/255.
  
                print(f"[DEBUG] {img.shape}")          
                np.save(f'./shapes3d_split/{shape}_{obj_color}_{bg_color}_{scale}.npy', img)
                if len(iid_test) == 0:
                    iid_test = test_img
                else:
                    iid_test = np.append(iid_test, test_img, axis = 0)
                tmp = [shape] * 5
                label_test.append(tmp)

label_test = np.reshape(label_test, -1)
print(f"[DEBUG] {iid_test.shape}")
print(f"[DEBUG] {label_test.shape}")
np.save(f'./shapes3d_split/iid_test.npy', iid_test)
np.save(f'./shapes3d_split/label_test.npy', label_test)
