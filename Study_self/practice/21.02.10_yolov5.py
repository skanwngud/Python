import torchvision
print(torchvision.__version__) # 0.8.2
import torch
print(torch.__version__) # 1.7.1

import numpy as np

from PIL import Image

from glob import glob

img_list=list()
for i in range(1, 76):
    img=glob('c:/data/image/project/train/face (%s).jpg'%i)
    img_list.append(img)

label_list=list()
for i in range(1, 34):
    label=glob('c:/data/image/project/label/face_label (%s).jpg'%i)
    label_list.append(label)

# print(len(img_list)) # 75
# print(len(label_list)) # 33

from sklearn.model_selection import train_test_split

train_img_list, val_img_list=train_test_split(
    img_list,
    train_size=0.8,
    random_state=23
)

# print(len(train_img_list), len(val_img_list)) # 60, 15

import yaml
with open('c:/data/image/project/'):
    

with open('c:/data/image/project/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list)+'\n')