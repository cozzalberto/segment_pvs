#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.models as models
import datetime
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
import torch
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.distributed as dist
import os
from PIL import Image
from torch.utils.data import Dataset
from datetime import timedelta
from torchsummary import summary
import sklearn
from util.util import EarlyStopper, MetricLogger, compute_scores, init_write, get_parser, replace_activation, get_warmup_then_steplr
from dataset.dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2 # If using PyTorch
import cv2
from torch.utils.data import random_split


current_time = datetime.datetime.now()

formatted_time = current_time.strftime("%Y%m%d-%H%M%S_%f")

file_path = f'logs/ablationstudy_{formatted_time}.txt'


def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120), rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()


# Define transformations (you can use your existing transformations)
transform = A.Compose([
    A.Resize(height=384, width=384, interpolation=1, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                   max_pixel_value=255.0, normalization="standard", p=1.0),
    ToTensorV2(p=1.0), 
])

training_transform = A.Compose([
    A.Resize(height=384, width=384, interpolation=1, p=1),
    A.HorizontalFlip(p=0.5),                  # Step 2: Basic geometric
    A.OneOf([
        A.ToGray(p=1.0),
        A.ChannelDropout(p=1.0)
    ], p=0.15),
    A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9),
                 angle_range=[0.1,0.5],
                 num_flare_circles_range=[2,8],
                 src_radius=100, src_color=(255, 255, 255),
                 p=0.2),
    A.SquareSymmetry(p=0.5),
    A.PlanckianJitter(p=0.3),
    A.RandomShadow(p=0.15),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                   max_pixel_value=255.0, normalization="standard", p=1.0),
    ToTensorV2(p=1.0),

])

args = get_parser()
batch_size = args.batch_size
num_epochs_phase1 = args.epochs_phase1
num_epochs_phase2 = args.epochs_phase2
learning_rate1 = args.lr_1
fine_tune_lr = args.lr_finetune
learning_rate_2 = fine_tune_lr
learning_diff = args.lr_diff
wd1 = args.wd1
wd2 = args.wd2
num_steplr = args.num_steplr
scheduler1 = args.scheduler1   #SCHEDULER1 È QUELLO DEL BACKBONE, SCHEDULER_DIFF È QUELLO DEL CLASSIFIER. STESSO CRITERIO PER I LEARNING RATE. ANCHE SE NUM_UNFROZEN1 È 0 È COSÌ!!!!
scheduler_diff = args.scheduler_diff
gamma_onecycle = 40
scheduler2 = args.scheduler2
num_unfrozen1 = args.num_unfrozen1
num_unfrozen2 = args.num_unfrozen2
smoothing = args.label_smoothing
optimizer_name =args.optimizer
t0 = args.t0
t_mult = args.t_mult 
milestones = args.milestones
model_name =args.model_name
activation = args.activation
if 'relu' in activation.lower() and 'prelu' not in activation.lower():
    activation = nn.ReLU
elif 'tanh' in activation.lower():
    activation= nn.Tanh
elif 'gelu' in activation.lower():
    activation= nn.GELU
elif 'prelu' in activation.lower():
    activation = nn.PReLU
elif 'sigmoid' in activation.lower():
    activation = nn.Sigmoid

#model_name = "models/efficientnet_weights.pth"
#dataset_filtered = CustomDataset(root_dir='dataset/big_dataset/train', classes=['positive', 'negative'], transform=None)
#indices = np.arange(len(dataset_filtered))
#my_labels = [dataset_filtered.labels[i] for i in indices]  
#train_idx, val_idx = train_test_split(indices, test_size = 0.15, random_state = 42, stratify = my_labels)
#trainset_filtered = Subset(CustomDataset(root_dir='dataset/big_dataset/train', classes=['positive', 'negative'], transform=training_transform), train_idx)
#valset = Subset(CustomDataset(root_dir='dataset/big_dataset/train', classes=['positive', 'negative'], transform = transform), val_idx)
# You can then create a DataLoader from this datase
train_path = '/leonardo_work/PHD_cozzani/seg_solarbackup/dataset/danish_dataset1/gentofte_trainval/train'
trainset_filtered = CustomDataset(root_dir='/leonardo_work/PHD_cozzani/seg_solarbackup/dataset/danish_dataset1/gentofte_trainval/train', classes=['positive', 'negative'], transform=training_transform)
valset = CustomDataset(root_dir='/leonardo_work/PHD_cozzani/seg_solarbackup/dataset/danish_dataset2/danish_with_bbr_and_google/gentofte_trainval/val', classes=['positive', 'negative'], transform = transform)
#trainloader = torch.utils.data.DataLoader(trainset_filtered, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)      
def train():
    rank, world_size, local_rank= setup()
    torch.cuda.set_device(local_rank)
    if "v2_l" in model_name:
        model = models.efficientnet_v2_l()
    elif "v2_m" in model_name:
        model = models.efficientnet_v2_m()
    elif "b0" in model_name:
        model = models.efficientnet_b0()
    else:
        model = models.efficientnet_v2_s()
    if model_name != 'f':
        model.load_state_dict(torch.load(model_name, map_location=f"cuda:{local_rank}"))
    #model.avgpool = nn.AdaptiveAvgPool2d((2,2))
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs,2)
                )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if 'silu' not in args.activation.lower():
        replace_activation(model, nn.SiLU, activation)
    model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # Dataset + Sampler
    dataset = trainset_filtered
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    valloader_local = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4) # DataLoader per la validazione
    accuracy_history = []
    loss_history = []
    loss_val_history=[]
    accuracy_history=[]
    precision_history = []
    recall_history = []
    class_weights=sklearn.utils.class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(trainset_filtered.labels),
        y = trainset_filtered.labels)
#    class_weights=torch.tensor([1,1],dtype=torch.float).to(local_rank)
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(local_rank)
    print(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights,label_smoothing=smoothing)
    stopper=EarlyStopper(patience=5)
    #scheduler = OneCycleLR(optimizer, 0.001, total_steps=None, epochs=num_epochs, steps_per_epoch=len(dataloader))
    #scheduler = StepLR(optimizer,step_size=10,gamma=0.1)
    logger = MetricLogger(local_rank, world_size)
    print(len(dataloader))
    for param in ddp_model.module.parameters():
        param.requires_grad = False

    # Assicurati che i parametri del classificatore siano sbloccati
    for param in ddp_model.module.classifier.parameters():
        param.requires_grad = True
 
    for param in ddp_model.module.features[-num_unfrozen1:].parameters():
        param.requires_grad = True
    
    backbone_params = list(filter(lambda p: p.requires_grad, ddp_model.module.features.parameters()))
    classifier_params = list(filter(lambda p: p.requires_grad, ddp_model.module.classifier.parameters()))
    # Definisci l'ottimizzatore per i parametri che richiedono il gradiente (solo il classificatore inizialmente)

    if num_unfrozen1>0:
        if 'adamw' in optimizer_name.lower():
            optim_classifier = optim.AdamW(classifier_params, lr=learning_diff, weight_decay=wd1)
            optim_backbone = torch.optim.AdamW(backbone_params, lr=learning_rate1,weight_decay=wd1)
        elif 'adam' in optimizer_name.lower():
            optim_classifier = optim.Adam(classifier_params, lr=learning_diff, betas = (args.beta1,args.beta2), weight_decay=wd1)
            optim_backbone = torch.optim.Adam(backbone_params, lr=learning_rate1,betas = (args.beta1,args.beta2), weight_decay=wd1)
        elif 'rmsprop' in optimizer_name.lower():
            optim_classifier = optim.RMSprop(classifier_params, lr=learning_diff, weight_decay=wd1)
            optim_backbone = torch.optim.RMSprop(backbone_params, lr=learning_rate1,weight_decay=wd1)
        elif 'sgd' in optimizer_name.lower():
            optim_classifier= optim.SGD(classifier_params, lr=10*learning_diff, momentum=0.9, weight_decay=wd1)
            optim_backbone = torch.optim.SGD(backbone_params, lr=10*learning_rate1, momentum=0.9, weight_decay=wd1)

        if scheduler1.lower()=='steplr':
            scheduler_backbone = StepLR(optim_backbone, step_size = num_steplr, gamma =0.1)
        elif scheduler1.lower()=='cosine':
            scheduler_backbone = CosineAnnealingLR(optim_backbone, T_max = num_epochs_phase1)
        elif scheduler1.lower()=='reduce':
            scheduler_backbone = ReduceLROnPlateau(optim_backbone, factor = 0.1,  patience =4)
        elif scheduler1.lower() =='warm_cosine':
            scheduler_backbone = CosineAnnealingWarmRestarts(optim_backbone, T_0 = t0, T_mult = t_mult)
        elif scheduler1.lower() =='multistep':
            scheduler_backbone = MultiStepLR(optim_backbone, milestones, 0.1)
        elif scheduler1.lower()=='warm_steplr':
            scheduler_backbone= get_warmup_then_steplr(optim_backbone, num_warmup_steps=len(dataloader)*5,      # Warmup per ~1.5 epoche
                                   step_size=len(dataloader) * 3,  # Decay ogni 5 epoche
                                   gamma=0.96                   # Dimezza lr ogni 5 epoche
                                   )


        else:
            raise ValueError("learning rate scheduler not recognized")

        if scheduler_diff.lower()=='steplr':
            scheduler_classifier = StepLR(optim_classifier, step_size=num_steplr, gamma=0.1)
        elif scheduler_diff.lower() == 'cosine':
            scheduler_classifier =CosineAnnealingLR(optim_classifier, T_max = num_epochs_phase1)
        elif scheduler_diff.lower() == 'reduce':
            scheduler_classifier = ReduceLROnPlateau(optim_classifier, factor = 0.1, patience = 4)
        elif scheduler_diff.lower() =='warm_cosine':
            scheduler_classifier = CosineAnnealingWarmRestarts(optim_classifier, T_0 = t0, T_mult = t_mult)
        elif scheduler_diff.lower() =='multistep':
            scheduler_classifier = MultiStepLR(optim_classifier, milestones, 0.1)
        elif scheduler_diff.lower() == 'warm_steplr':
             scheduler_classifier= get_warmup_then_steplr(optim_classifier, num_warmup_steps=len(dataloader)*5,      # Warmup per ~1.5 epoche
                                   step_size=len(dataloader) * 3,  # Decay ogni 5 epoche
                                   gamma=0.96                   # Dimezza lr ogni 5 epoche
                                   )



        else:
            raise ValueError("learning rate scheduler not recognized")
    else:
        optim_classifier = optim.AdamW(classifier_params, lr=learning_diff, weight_decay=wd1)

        if scheduler_diff.lower()=='steplr':
            scheduler_classifier = StepLR(optim_classifier, step_size=num_steplr, gamma=0.1)
        elif scheduler_diff.lower() == 'cosine':
            scheduler_classifier =CosineAnnealingLR(optim_classifier, T_max = num_epochs_phase1)
        elif scheduler_diff.lower() == 'reduce':
            scheduler_classifier = ReduceLROnPlateau(optim_classifier, factor = 0.6, patience = 3)
        elif scheduler_diff.lower() =='warm_cosine':
            scheduler_classifier = CosineAnnealingWarmRestarts(optim_classifier, T_0 = t0, T_mult = t_mult)
        elif scheduler_diff.lower() =='multistep':
            scheduler_classifier = MultiStepLR(optim_classifier, milestones, 0.1)

        else:
            raise ValueError("learning rate scheduler not recognized")

    #Definisci l'ottimizzatore per i parametri che richiedono il gradiente (solo il classificatore inizialmente)   
    #17 # Addestra solo il classificatore per 5 epoche (puoi sperimentare)
    if rank == 0:
        init_write(file_path, batch_size, num_epochs_phase1, num_epochs_phase2, learning_rate_2, wd1, wd2, scheduler1, scheduler2, scheduler_diff,  num_unfrozen1, num_unfrozen2,num_steplr, gamma_onecycle, learning_diff, learning_rate1 = learning_rate1 )
        with open(file_path,'a') as f:
            f.write(f"{train_path}\n smoothing {smoothing}\n activation {activation}\n t0 (cosine):{t0}  T_mult {t_mult}\n milestones (multistep): {milestones}\n {model_name}\n {optimizer_name}\n beta1 {args.beta1} beta2 {args.beta2}\n")

    for epoch in range(num_epochs_phase1):
        sampler.set_epoch(epoch)
        ddp_model.train() # Imposta il modello in modalità training
        running_loss = 0.0
        total_samples = 0
        stop_flag = torch.tensor(False, device = local_rank)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(local_rank), labels.to(local_rank)
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            if num_unfrozen1>0:
                optim_backbone.zero_grad()
            optim_classifier.zero_grad()
            loss.backward()
            if num_unfrozen1>0:
                optim_backbone.step()
            optim_classifier.step()
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            if scheduler1.lower()=='warm_steplr':
                scheduler_backbone.step()
            if scheduler_diff.lower()=='warm_steplr':
                scheduler_classifier.step()
        dist.barrier() # Sincronizza i processi DDP

        train_loss = running_loss / total_samples
        val_acc, val_precision, val_recall, val_loss = compute_scores(ddp_model, valloader_local, local_rank)
        ddp_model.train() # Riporta il modello in modalità training dopo la validazione

        val_loss_avg = logger.get_global_metric(val_loss)
        val_recall_avg = logger.get_global_metric(val_recall)
        val_precision_avg = logger.get_global_metric(val_precision)
        train_loss_avg = logger.get_global_metric(train_loss)
        if num_unfrozen1>0:
            if scheduler1.lower()!='reduce' and scheduler1.lower()!='warm_steplr':
                scheduler_backbone.step()
            elif scheduler1.lower()=='reduce':
                scheduler_backbone.step(val_loss_avg)
        if scheduler_diff.lower()!= 'reduce' and scheduler_diff.lower()!='warm_steplr':
            scheduler_classifier.step()
        elif scheduler_diff.lower()=='reduce':
            scheduler_classifier.step(val_loss_avg)
        if rank == 0:
            loss_history.append(train_loss_avg)
            accuracy_history.append(val_acc)
            precision_history.append(val_precision_avg)
            recall_history.append(val_recall_avg)
            loss_val_history.append(val_loss_avg)
            print(f"[Fase 1 - Epoch {epoch+1}/{num_epochs_phase1}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss_avg:.4f} Val Acc: {val_acc:.4f} Prec: {val_precision_avg:.4f} Recall: {val_recall_avg:.4f}")
            if num_unfrozen1>0:
                with open(file_path,'a') as f:
                    f.write(f"[Fase 1 - Epoch {epoch+1}/{num_epochs_phase1}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss_avg:.4f} Val Acc: {val_acc:.4f} Prec: {val_precision_avg:.4f} Recall: {val_recall_avg:.4f} LR backbone: {scheduler_backbone.get_last_lr()[0]} LR classifier: {scheduler_classifier.get_last_lr()[0]} \n")
            else:
                with open(file_path,'a') as f:
                    f.write(f"[Fase 1 - Epoch {epoch+1}/{num_epochs_phase1}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss_avg:.4f} Val Acc: {val_acc:.4f} Prec: {val_precision_avg:.4f} Recall: {val_recall_avg:.4f}  LR classifier: {scheduler_classifier.get_last_lr()[0]} \n")

            should_stop, best_loss, epochs_no_improvement = stopper.early_stopping(val_loss_avg)
            if should_stop:
                print(f'early stopping in phase1 at epoch {epoch+1}\n')
                stop_flag = True
                stop_flag = torch.tensor(stop_flag, device = local_rank)
        torch.distributed.broadcast(stop_flag, src = 0)
        if stop_flag.item():
            break

    for param in ddp_model.module.parameters():
        param.requires_grad = False

    # Assicurati che i parametri del classificatore siano sbloccati
    for param in ddp_model.module.classifier.parameters():
        param.requires_grad = True


    for param in ddp_model.module.features[-num_unfrozen2:].parameters():
        param.requires_grad = True # Ora tutti i parametri saranno addestrabili



    # Re-inizializza l'ottimizzatore con il nuovo learning rate e i parametri aggiornati
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=fine_tune_lr, weight_decay=wd2)
    # Puoi anche usare OneCycleLR qui per un LR che varia dinamicamente
    if num_epochs_phase2>0:
        if scheduler2.lower()=='steplr':
            scheduler= StepLR(optimizer, step_size = num_steplr, gamma =0.1)
        elif scheduler2.lower()=='cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max = num_epochs_phase2)
        elif scheduler2.lower()=='onecycle':
            scheduler = OneCycleLR(optimizer, max_lr=fine_tune_lr *gamma_onecycle , epochs=num_epochs_phase2, steps_per_epoch=len(dataloader))
        elif scheduler2.lower()=='reduce':
            scheduler = ReduceLROnPlateau(optimizer, "min", factor = 0.5, patience = 3)
        elif scheduler2.lower() =='warm_cosine':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = t0, T_mult = t_mult)
        elif scheduler2.lower() =='multistep':
            scheduler = MultiStepLR(optimizer, milestones, 0.1)

        else:
            raise ValueError("learning rate scheduler not recognized")
        #scheduler = StepLR(optimizer,step_size= 3,gamma=0.1)
    stopper_phase2 = EarlyStopper(patience = 3)
    if rank == 0:
        print("\n--- Fase 2: Fine-tuning dell'intera rete (o sezioni) ---")

    for epoch in range(num_epochs_phase2):
        sampler.set_epoch(epoch + num_epochs_phase1) # Continua con l'indice di epoca corretto
        ddp_model.train()
        running_loss = 0.0
        total_samples = 0
        stop_flag = torch.tensor(False, device = local_rank)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(local_rank), labels.to(local_rank)
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler2.lower() == 'onecycle': scheduler.step() # Step del scheduler dopo ogni batch
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        if scheduler2.lower() != 'onecycle' and scheduler2.lower() !='reduce':
            scheduler.step()
        dist.barrier()

        train_loss = running_loss / total_samples
        val_acc, val_precision, val_recall, val_loss = compute_scores(ddp_model, valloader_local, local_rank)
        ddp_model.train()
        val_loss_avg = logger.get_global_metric(val_loss)
        val_recall_avg = logger.get_global_metric(val_recall)
        val_precision_avg = logger.get_global_metric(val_precision)
        train_loss_avg = logger.get_global_metric(train_loss)
        if scheduler2.lower() == 'reduce': scheduler.step(val_loss_avg)

        
        if rank == 0:
            loss_history.append(train_loss_avg)
            accuracy_history.append(val_acc)
            precision_history.append(val_precision_avg)
            recall_history.append(val_recall_avg)
            loss_val_history.append(val_loss_avg)
            print(f"[Fase 2 - Epoch {epoch+1}/{num_epochs_phase2}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss_avg:.4f} Val Acc: {val_acc:.4f} Prec: {val_precision_avg:.4f} Recall: {val_recall_avg:.4f} Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            with open(file_path,'a') as f:
                f.write(f"[Fase 2 - Epoch {epoch+1}/{num_epochs_phase2}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss_avg:.4f} Val Acc: {val_acc:.4f} Prec: {val_precision_avg:.4f} Recall: {val_recall_avg:.4f} Current LR: {optimizer.param_groups[0]['lr']:.6f}\n")
            should_stop, best_loss, epochs_no_improvement = stopper_phase2.early_stopping(val_loss_avg)
            if should_stop:
                print(f'early stopping in phase2 at epoch {epoch}\n')
                stop_flag = True 
                stop_flag = torch.tensor(stop_flag, device = local_rank)
        torch.distributed.broadcast(stop_flag, src = 0)
        if stop_flag.item():
            break

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), f"logs/nofreeze_{formatted_time}_bs{batch_size}.pth")
        with open(file_path,'a') as f:
            print("Model saved after training nicee",file=f)
        fig, axs= plt.subplots(1,2,figsize=(20,10))
        axs[0].plot(loss_history,label='Train Loss', marker='o', linewidth=2)
        axs[0].plot(loss_val_history,label='Val Loss', marker='s', linewidth=2)
        axs[0].legend(['train loss', 'val loss'])
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[1].plot(precision_history, label='Precision', marker='o', linewidth=2)
        axs[1].plot(recall_history,label='Recall', marker='s', linewidth=2)
        axs[1].legend(['Precision', 'Recall'])
        axs[1].set_title('Precision and Recall')
        plt.tight_layout()
        plt.savefig(f'logs/metrics_{formatted_time}.png')
        plt.show()
        cleanup()
 
   

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    train()


