import os
import shutil
import random

input_dir='big_dataset/train'
train_path = 'big_dataset2/train'
val_path ='big_dataset2/val'
for split in ['train', 'val']:
    for subfolder in ['positive', 'negative']:
        os.makedirs(os.path.join('big_dataset2', split, subfolder), exist_ok=True)

input_list_positive = os.listdir(os.path.join(input_dir,'positive'))
for file_name in input_list_positive:
    basename = os.path.splitext(file_name)[0]
    file_path = os.path.join(input_dir,'positive', file_name)
    #mask_file = basename + '.png'
    #mask_path = os.path.join(input_dir,'mask', mask_file)
    if random.random() < 0.85:
        shutil.copyfile(file_path, os.path.join(train_path, 'positive', file_name))
     #   shutil.copyfile(mask_path, os.path.join(train_path, 'mask', mask_file))
    else :
        shutil.copyfile(file_path, os.path.join(val_path, 'positive', file_name))
     #   shutil.copyfile(mask_path, os.path.join(val_path, 'mask', mask_file))

input_list_negative = os.listdir(os.path.join(input_dir,'negative'))
for file_name in input_list_negative:
    basename = os.path.splitext(file_name)[0]
    file_path = os.path.join(input_dir, 'negative', file_name)
    if random.random() < 0.85:
        shutil.copyfile(file_path, os.path.join(train_path, 'negative', file_name))
    else :
        shutil.copyfile(file_path, os.path.join(val_path, 'negative', file_name))

