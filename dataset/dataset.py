import os 
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import Dataset as hf_Dataset
import cv2
import albumentations as A
import shutil

class CustomDataset(Dataset):                                #per training classifier 
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(classes):
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image = image)
            image = augmented['image']
        return image, label


class CustomImageDataset(Dataset):
    def __init__(self,img_dir, transform=None):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg','.tif'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]



def LabelMap(t):
  t[t==255] =1
  return t


def load_image_mask_pairs(image_dir, mask_dir):
    image_filenames = sorted(os.listdir(image_dir))
    data = []

    for fname in image_filenames:
        root_name = os.path.splitext(fname)[0]
        image_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, root_name + '.png')

        # Ensure mask exists
        if not os.path.exists(mask_path):
            continue

        data.append({
            "image": image_path,
            "segmentation_mask": mask_path
        })

    return hf_Dataset.from_list(data)

def load_image_label_pairs(root_dir,labels):
    data = []
    for class_idx, label in enumerate(labels):
        image_dir = os.path.join(root_dir,label)
        image_filenames = sorted(os.listdir(image_dir))
        for fname in image_filenames:
            image_path = os.path.join(image_dir, fname)
            data.append({
                "image": image_path,
                "label": class_idx
            })

    return hf_Dataset.from_list(data)


from random import sample

def add_files(N, input_dir, output_dir):
    filenames=[filename for filename in os.listdir(input_dir)]   
    filenames_da_inserire = sample(filenames,N)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Directory already exists, files will be added to it.")
    print("Adding {} files to {}".format(len(filenames_da_inserire), output_dir))
    # Copy the selected files to the output directory
    for filename in filenames_da_inserire:
        src = os.path.join(input_dir,filename)
        dst= os.path.join(output_dir,filename)
        if not os.path.exists(dst):
            shutil.move(src, output_dir)        

def add_files_and_masks(N, input_dir, output_dir):
    filenames=[filename for filename in os.listdir(os.path.join(input_dir, 'positive'))]
    filenames_da_inserire = sample(filenames,N)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Directory already exists, files will be added to it.")
    output_img = os.path.join(output_dir, 'positive')
    output_mask = os.path.join(output_dir, 'mask')
    if not os.path.exists(output_img):
        os.makedirs(output_img)
    else:
        print("Directory for images already exists, files will be added to it.")

    if not os.path.exists(output_mask):
        os.makedirs(output_mask)
    else:
        print("Mask directory already exists, files will be added to it.")

    print("Adding {} files to {}".format(len(filenames_da_inserire), output_dir))
    # Copy the selected files to the output directory
    for filename in filenames_da_inserire:
        shutil.move(os.path.join(input_dir, 'positive', filename),output_img)
        shutil.move(os.path.join(input_dir, 'mask', filename), output_mask)
        print(f' file {filename} added to {output_img}')

