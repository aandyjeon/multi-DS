import os
import numpy as np
import cv2

from itertools import combinations, product
from PIL import Image

file_path = './Anno_fine/list_attr_cloth.txt'
attr_name_dir = './Anno_fine/list_attr_img.txt'

def load_attributes(file_path):
    attributes = {}
    with open(file_path, 'r') as f:
        next(f)
        next(f)
        i = 0
        for line in f:
            parts = line.strip().split()
            attribute_name = ' '.join(parts[:-1])
            attribute_type = i
            attributes[attribute_name] = attribute_type
            i += 1
    return attributes

def create_image_attributes_dict(file_content):
    lines = file_content.strip().split("\n")
    
    image_attributes_dict = {}
    
    for line in lines[2:]:  
        parts = line.split()
        image_path = parts[0]
        attributes = [int(attr) for attr in parts[1:]]

        image_attributes_dict[image_path] = attributes
    
    return image_attributes_dict

def read_and_process_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    return create_image_attributes_dict(file_content)

def transform_image_path(old_path):
    _, tail = os.path.split(old_path)
    category, filename = tail.rsplit('-', 1)
    new_path = f"./img/{category}/{filename}"
    
    return new_path

def count_combinations_for_set(lists, combination_counts, img_list, img_path, attr_list):
    indices = [lst.index(1) + 1 if 1 in lst else 0 for lst in lists]  # Determine indices of 1s
    if all(index > 0 for index in indices):  # Ensure all lists have a 1
        img_list.append(Image.open(img_path))
        attr_list.append(indices)
        combination_counts[tuple(indices)] += 1
    
    return img_list, attr_list

attribute_dict = load_attributes(file_path)
image_attributes_dict = read_and_process_file(attr_name_dir)

a = [0, 5]
b = [7, 8]
c = [18, 19]
d = [11, 12]

comb_a = list(combinations(a, 2))
comb_b = list(combinations(b, 2))
comb_c = list(combinations(c, 2))
comb_d = list(combinations(d, 2))

all_combinations = list(product(comb_a, comb_b, comb_c, comb_d))

imgs = []
attrs = []

for combination in all_combinations:
    comb_a, comb_b, comb_c, comb_d = combination

    combination_counts = {(i, j, k, l): 0 for i in range(1, 3) for j in range(1, 3) for k in range(1, 3) for l in range(1, 3)}

    for img, attr in image_attributes_dict.items():
        image_path = transform_image_path(img)
        attr_a = [attr[i] for i in comb_a]
        attr_b = [attr[i] for i in comb_b]
        attr_c = [attr[i] for i in comb_c]
        attr_d = [attr[i] for i in comb_d]

        lists = [attr_a, attr_b, attr_c, attr_d]
        imgs, attrs = count_combinations_for_set(lists, combination_counts, imgs, image_path, attrs)

a = [1, 2]
b = [1, 2]
c = [1, 2]
d = [1, 2]
combs = {}

for _a in a:
    for _b in b:
        for _c in c:
            for _d in d:
                comb_img = []
                
                for img, attr in zip(imgs, attrs):
                    if attr == [_a, _b, _c, _d]:
                        comb_img.append(np.array(img))
                
                if comb_img:
                    combs[(_a, _b, _c, _d)] = comb_img

test_img = []
for k, v in combs.items():
    """
    a = [0, 5]
    b = [7, 8]
    c = [18, 19]
    d = [11, 12]
    """
    _a, _b, _c, _d = k
    if _a == 1: _a = 0
    else: _a = 5
    if _b ==1: _b = 7
    else: _b = 8
    if _c ==1: _c = 18
    else: _c = 19
    if _d ==1: _d = 11
    else: _d = 12
    resized_imgs = []
    for img in v:
        resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        resized_imgs.append(resized_img)
    
    train_output = resized_imgs[:-5]
    test_output  = resized_imgs[-5:]

    np.save(f'./deepfashion_split/{_d}_{_a}_{_b}_{_c}.npy', np.array(train_output)/255.)
    np.save(f'./deepfashion_split/test_{_d}_{_a}_{_b}_{_c}.npy', np.array(test_output)/255.)
