#!/usr/bin/env python

from torchvision.utils import draw_segmentation_masks
from skimage.measure import regionprops, label
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import SegformerImageProcessor
from transformers import Trainer, SegformerForSemanticSegmentation, TrainingArguments
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from pyproj import Transformer
from torchvision.io.image import decode_image
from torchvision.transforms.functional import to_pil_image
import random


def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean

def show(imgs,path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

#create output directory
if not os.path.exists('logs/predicted_masks15Agosto'):
    os.makedirs('logs/predicted_masks15Agosto')
img_with_masks_path = 'logs/images_and_masks15Agosto'

if not os.path.exists(img_with_masks_path):
    os.makedirs(img_with_masks_path)

with open('centroids.txt', 'w') as f:
    f.write("Latitude Longitude Area\n")

class_to_idx ={'background': 0, 'solar_panel':1}

transformer = Transformer.from_crs("epsg:32632", "epsg:4326", always_xy=True)

#feature_extractor = SegformerImageProcessor.from_pretrained("models/segformer", size ={'height': 320,'width':320}, local_files_only=True)  # o tuo config.json custom
feature_extractor = SegformerImageProcessor.from_pretrained("models/models--nvidia--mit-b2/snapshots/mit-b2", size ={'height': 320,'width':320}, local_files_only=True)  # o tuo config.json custom
# Carica il modello da checkpoint
model = SegformerForSemanticSegmentation.from_pretrained(
    "logs/segformer_finetuning2_20250813105537/checkpoint-6752",  # <-- cambia con il percorso giusto
    local_files_only=True
)

class test_dataset(Dataset):
    def __init__(self,test_dir, processor = None):
        self.test_dir = test_dir
        self.test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png','.tif')) ]
        self.processor = processor
    def __len__(self):
        return len(self.test_files)

    def __getitem__(self,idx):
        test_filepath = os.path.join(self.test_dir, self.test_files[idx])
        image = Image.open(test_filepath).convert('RGB')
        inputs = self.processor(images=image, return_tensors = 'pt')
        pixel_values = inputs['pixel_values'].squeeze(0)
        return pixel_values, self.test_files[idx]
        
test_ds = test_dataset('/leonardo_work/PHD_cozzani/seg_solar/dataset/positives_15Agosto', processor= feature_extractor)
test_loader = DataLoader(test_ds, batch_size =16, shuffle = False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


for batch, file_path in test_loader:
    batch = batch.to(device)
    with torch.no_grad():
        outputs = model(batch)
    logits = outputs.logits
    logits = F.interpolate(
                           logits,
                           size = 320,
                           mode = 'nearest')
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    for l, prediction in enumerate(predictions):
        save = False
        file_name = os.splitex(file_path[l])[0]
        regions = regionprops(label(prediction))
        prediction = label(prediction)
        for region in regions:
            if region.area < 450:  # Filtra le regioni piccole
                prediction[prediction == region.label] = 0
            else:
                save = True
                row, col = region.centroid
                image = os.path.join('/leonardo_work/PHD_cozzani/seg_solar/dataset/positives_15Agosto', file_path[l])
                with rasterio.open(image) as dataset:
                    x_geo, y_geo = dataset.xy(row, col)
                    #print(dataset.crs)
                    lon, lat = transformer.transform(x_geo, y_geo)      
                    with open('centroids.txt', 'a') as f:
                        f.write(f"{lat} {lon} {region.area}\n")  # Salva le coordinate del centroide
    # Salva le maschere filtrate
            if save and random.random() < 1.02:  # Salva solo il 10% delle maschere filtrate
                pred_image = Image.fromarray(prediction.astype(np.uint8) * 255)  # Converti in immagine binaria
                output_path = os.path.join('logs/predicted_masks15Agosto',f"{file_name}_mask_{l}.png")  # Cambia il percorso di output come necessario
                pred_image.save(output_path)
#                print(f"Saved {output_path}")
                class_dim = 1
                boolean_train_mask = torch.from_numpy(prediction != class_to_idx['background'])
                #img_height, img_width = img.shape[-2:]
                # boolean_train_mask needs to be treated as a batch of 1 mask for interpolation
                #boolean_solar_mask_resized = F.interpolate(
                #    boolean_train_mask.unsqueeze(1).float(), # Add channel and batch dimensions, convert to float for interpolation
                #    size=(img_height, img_width),
                #    mode='nearest' # Use nearest neighbor for boolean masks
                #).squeeze(1).bool()
                train_with_masks = [
                    draw_segmentation_masks(unnormalize(batch[l]), masks=boolean_train_mask, alpha=0.3,colors='green')
                ]
                show(train_with_masks,os.path.join(img_with_masks_path, f"{file_name}_mask.png")) 
                
# Salva le maschere filtrate    


