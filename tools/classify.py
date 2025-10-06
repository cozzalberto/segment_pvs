#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataset.dataset import CustomImageDataset
from util.util import unnormalize, copy_positive
import shutil

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384,384)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


model = models.efficientnet_v2_s()
model.load_state_dict(torch.load("models/efficientnet_weights.pth", map_location=device))
model.avgpool = nn.AdaptiveAvgPool2d((2,2))
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs*2*2,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,2),
)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(device)
model.load_state_dict(torch.load("logs/nofreeze_20250811183103.pth", map_location=device))
model.eval()

test_set=CustomImageDataset(img_dir='/leonardo_work/PHD_cozzani/seg_solar/dataset/dataset_true/',  transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=8,shuffle=False)

output_dir = '/leonardo_work/PHD_cozzani/seg_solar/dataset/positives_15Agosto'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
classes=['positive','negative']
copy_positive(model, test_loader, classes, device,  output_dir)




