from dataset.dataset import CustomDataset
from util.util import EarlyStopper, MetricLogger, compute_scores, init_write, compute_scores_seg
import albumentations as A
from albumentations.pytorch import ToTensorV2 # If using PyTorch
import cv2
import torch 
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import datetime
import torch.nn as nn
from transformers import SegformerForImageClassification, SegformerImageProcessor

current_time = datetime.datetime.now()

formatted_time = current_time.strftime("%Y%m%d%H%M%S")
model_name = "logs/segclass_20251001094913.pth"

if not os.path.exists('logs/test'):
    os.makedirs('logs/test', exist_ok = True)

file_path =f'logs/test/testseg_{formatted_time}'

bologna = True
herlev = True
segformer = True
vgg= False
efficient = False
s = False
if efficient == True and s == False:
    m = True
if efficient:
    transform = A.Compose([
        A.Resize(height=384, width=384, interpolation=1, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                       max_pixel_value=255.0, normalization="standard", p=1.0),
        ToTensorV2(p=1.0),
    ])
elif vgg:
    transform = A.Compose([
    A.Resize(height=224, width=224, interpolation=1, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                   max_pixel_value=255.0, normalization="standard", p=1.0),
    ToTensorV2(p=1.0),
    ])

else:
     transform = A.Compose([
        A.Resize(height=512, width=512, interpolation=1, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                       max_pixel_value=255.0, normalization="standard", p=1.0),
        ToTensorV2(p=1.0),
    ])

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if efficient:
    if s:
        model = models.efficientnet_v2_s()
        model.load_state_dict(torch.load("models/efficientnet_weights.pth", map_location=device))

    else:
        model = models.efficientnet_v2_m()
        model.load_state_dict(torch.load("models/efficientnet_v2_m-dc08266a.pth", map_location=device))
    
    model.avgpool = nn.AdaptiveAvgPool2d((2,2))
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_ftrs*2*2,512),
                nn.ReLU(),
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256,2),
        )
    model.to(device)
    model.load_state_dict(torch.load(model_name, map_location=device, weights_only= False))
elif vgg:
    model = models.vgg16()
    model.load_state_dict(torch.load("models/vgg16_weights.pth", map_location=device))
    model.avgpool = nn.AdaptiveAvgPool2d((2,2))
    print(model.classifier)
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512*2*2,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,2),
    )
    model.load_state_dict(torch.load(model_name, map_location=device, weights_only= False))
    model.to(device)
else:
    model = SegformerForImageClassification.from_pretrained(
    "models/segformer_offline")
    model.classifier = nn.Linear(model.classifier.in_features, 2)  # se avevi cambiato la testa
    model.load_state_dict(torch.load(model_name, map_location=device, weights_only = False))
    model.to(device)
    model.eval()
    


with open(file_path, 'w') as f:
    f.write(f'{model_name}\n')

if herlev:
    testpath = '/leonardo_work/PHD_cozzani/seg_solar/dataset/danish_dataset1/herlev_test/test/'
    testset = CustomDataset(testpath, classes=['positive', 'negative'], transform = transform)
    test_loader = DataLoader(testset, batch_size = 32, num_workers =8)
    if efficient or vgg:
        accuracy, precision, recall, val_loss = compute_scores(model, test_loader, device)
    else: 
        accuracy, precision, recall, val_loss = compute_scores_seg(model, test_loader, device)

    with open(file_path, 'a') as f:
        f.write(f'**test** Accuracy: {accuracy}, precision: {precision}, recall: {recall}, loss: {val_loss}\n')

if bologna:                                                                                                                              
    for test in ['merged', 'casalecchio', 'mengoli', 'suburbano2']:
        testpath = os.path.join('dataset/dataset_bolo_class/', test)
        testset= CustomDataset(testpath, classes=['positive', 'negative'], transform = transform)
        test_loader = DataLoader(testset, batch_size = 32, num_workers =8)
        if efficient or vgg:
            accuracy, precision, recall, val_loss = compute_scores(model, test_loader, device)
        else:
            accuracy, precision, recall, val_loss = compute_scores_seg(model, test_loader, device)

        with open(file_path, 'a') as f:
            f.write(f'**{test}** Accuracy: {accuracy}, precision: {precision}, recall: {recall}, loss: {val_loss}\n')
