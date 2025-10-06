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
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
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
from util.util import EarlyStopper, MetricLogger, compute_scores, init_write
from dataset.dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2 # If using PyTorch
import cv2
from torch.utils.data import random_split



current_time = datetime.datetime.now()

formatted_time = current_time.strftime("%Y%m%d%H%M%S")

file_path = f'logs/log_{formatted_time}.txt'


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
                 p=0.3),
    A.SquareSymmetry(p=0.5),
    A.PlanckianJitter(p=0.3),
    A.RandomShadow(p=0.15),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                   max_pixel_value=255.0, normalization="standard", p=1.0),
    ToTensorV2(p=1.0),

])

batch_size = 8
num_epochs_phase1 = 10
num_epochs_phase2 = 0 
learning_rate1 = 1e-4
fine_tune_lr = 1e-5
learning_rate_2 = fine_tune_lr
learning_diff = 1e-3
wd1 = 5e-4
wd2 = 5e-4
num_steplr = 10
scheduler1 ='cosine'   #SCHEDULER1 È QUELLO DEL BACKBONE, SCHEDULER_DIFF È QUELLO DEL CLASSIFIER. STESSO CRITERIO PER I LEARNING RATE. ANCHE SE NUM_UNFROZEN1 È 0 È COSÌ!!!!
scheduler_diff = 'steplr'
gamma_onecycle = 10
scheduler2 = 'onecycle'
num_unfrozen1 = 5 
num_unfrozen2 = 0 

#dataset_filtered = CustomDataset(root_dir='dataset/big_dataset/train', classes=['positive', 'negative'], transform=None)
#indices = np.arange(len(dataset_filtered))
#my_labels = [dataset_filtered.labels[i] for i in indices]  
#train_idx, val_idx = train_test_split(indices, test_size = 0.15, random_state = 42, stratify = my_labels)
#trainset_filtered = Subset(CustomDataset(root_dir='dataset/big_dataset/train', classes=['positive', 'negative'], transform=training_transform), train_idx)
#valset = Subset(CustomDataset(root_dir='dataset/big_dataset/train', classes=['positive', 'negative'], transform = transform), val_idx)
# You can then create a DataLoader from this dataset
trainset_filtered = CustomDataset(root_dir='dataset/danish_bbr_google_noherlev/gentofte_trainval/train', classes=['positive', 'negative'], transform=training_transform)
valset = CustomDataset(root_dir='dataset/danish_bbr_google_noherlev/gentofte_trainval/val', classes=['positive', 'negative'], transform = transform)
#trainloader = torch.utils.data.DataLoader(trainset_filtered, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)      
def train():
    rank, world_size, local_rank= setup()
    print(world_size)
    torch.cuda.set_device(local_rank)
    model = models.efficientnet_v2_s()
    model.load_state_dict(torch.load("models/efficientnet_weights.pth", map_location=f"cuda:{local_rank}"))
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
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(local_rank)

    print(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    stopper=EarlyStopper(patience=15)
    #scheduler = OneCycleLR(optimizer, 0.001, total_steps=None, epochs=num_epochs, steps_per_epoch=len(dataloader))
    #scheduler = StepLR(optimizer,step_size=10,gamma=0.1)
    logger = MetricLogger(local_rank, world_size)
    
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
        optimizer = optim.Adam([{'params':backbone_params,'lr':learning_rate1},{'params': classifier_params, 'lr':learning_diff}], weight_decay=wd1) 
        optim_backbone = torch.optim.Adam(backbone_params, lr=learning_rate1)
    else:
        optimizer = optim.Adam([{'params': classifier_params, 'lr':learning_diff}], weight_decay=wd1)

    optim_classifier = torch.optim.Adam(classifier_params, lr=learning_diff)
    #scheduler_backbone = CosineAnnealingLR(optim_backbone, T_max = num_epochs_phase1)
    if num_unfrozen1>0:
        if scheduler1.lower()=='steplr':
            scheduler_backbone = StepLR(optim_backbone, step_size = num_steplr, gamma =0.3)
        elif scheduler1.lower()=='cosine':
            scheduler_backbone = CosineAnnealingLR(optim_backbone, T_max = num_epochs_phase1)
        else:
            raise ValueError("learning rate scheduler not recognized")

        if scheduler_diff.lower()=='steplr':
            scheduler_classifier = StepLR(optim_classifier, step_size=num_steplr, gamma=0.1)
        elif scheduler_diff.lower() == 'cosine':
            scheduler_classifier =CosineAnnealingLR(optim_classifier, T_max = num_epochs_phase1)
        else:
            raise ValueError("learning rate scheduler not recognized")
    else:
        if scheduler_diff.lower()=='steplr':
            scheduler_classifier = StepLR(optimizer, step_size=num_steplr, gamma=0.1)
        elif scheduler_diff.lower() == 'cosine':
            scheduler_classifier =CosineAnnealingLR(optimizer, T_max = num_epochs_phase1)
        else:
            raise ValueError("learning rate scheduler not recognized")

    #Definisci l'ottimizzatore per i parametri che richiedono il gradiente (solo il classificatore inizialmente)   
    #17 # Addestra solo il classificatore per 5 epoche (puoi sperimentare)
    if rank == 0:
        init_write(file_path, batch_size, num_epochs_phase1, num_epochs_phase2, learning_rate_2, wd1, wd2, scheduler1, scheduler2, scheduler_diff,  num_unfrozen1, num_unfrozen2,num_steplr, gamma_onecycle, learning_diff, learning_rate1 = learning_rate1 )
        with open(file_path,'a') as f:
            f.write(file_path)

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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        if num_unfrozen1 >0:
            scheduler_backbone.step()
        scheduler_classifier.step()
        if num_unfrozen1 >0 :
            optimizer.param_groups[0]['lr'] = scheduler_backbone.get_last_lr()[0]
            optimizer.param_groups[1]['lr'] = scheduler_classifier.get_last_lr()[0]
        dist.barrier() # Sincronizza i processi DDP

        train_loss = running_loss / total_samples
        val_acc, val_precision, val_recall, val_loss = compute_scores(ddp_model, valloader_local, local_rank)
        ddp_model.train() # Riporta il modello in modalità training dopo la validazione

        val_loss_avg = logger.get_global_metric(val_loss)
        val_recall_avg = logger.get_global_metric(val_recall)
        val_precision_avg = logger.get_global_metric(val_precision)
        train_loss_avg = logger.get_global_metric(train_loss)
             
        if rank == 0:
            loss_history.append(train_loss_avg)
            accuracy_history.append(val_acc)
            precision_history.append(val_precision_avg)
            recall_history.append(val_recall_avg)
            loss_val_history.append(val_loss_avg)
            print(f"[Fase 1 - Epoch {epoch+1}/{num_epochs_phase1}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss_avg:.4f} Val Acc: {val_acc:.4f} Prec: {val_precision_avg:.4f} Recall: {val_recall_avg:.4f}")
            if num_unfrozen1>0:
                with open(f"logs/log_{formatted_time}.txt",'a') as f:
                    f.write(f"[Fase 1 - Epoch {epoch+1}/{num_epochs_phase1}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss_avg:.4f} Val Acc: {val_acc:.4f} Prec: {val_precision_avg:.4f} Recall: {val_recall_avg:.4f} LR backbone: {scheduler_backbone.get_last_lr()[0]} LR classifier: {scheduler_classifier.get_last_lr()[0]} \n")
            else:
                with open(f"logs/log_{formatted_time}.txt",'a') as f:
                    f.write(f"[Fase 1 - Epoch {epoch+1}/{num_epochs_phase1}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss_avg:.4f} Val Acc: {val_acc:.4f} Prec: {val_precision_avg:.4f} Recall: {val_recall_avg:.4f}  LR classifier: {scheduler_classifier.get_last_lr()[0]} \n")

            should_stop, best_loss, epochs_no_improvement = stopper.early_stopping(val_loss_avg)
            if should_stop:
                print(f'early stopping in phase1 at epoch {epoch+1}\n')
                stop_flag = True
                stop_flag = torch.tensor(stop_flag, device = local_rank)
        torch.distributed.broadcast(stop_flag, src = 0)
        if stop_flag.item():
            break



    # --- FASE 2: Fine-tuning dell'intera rete (o di strati selezionati) ---
    # Scongela tutti i parametri (o solo gli ultimi blocchi, vedi note sotto)
    for param in ddp_model.module.features[-num_unfrozen2:].parameters():
        param.requires_grad = True # Ora tutti i parametri saranno addestrabili

    # Per un fine-tuning più specifico, potresti scongelare solo gli ultimi blocchi:
    # Esempio per EfficientNet-B0: il modulo `features` contiene 8 blocchi convoluzionali.
    # Puoi scongelare gli ultimi 2-3 blocchi per un fine-tuning più mirato.
    # for i, block in enumerate(ddp_model.module.features):
    #     if i >= 6: # Scongela gli ultimi 2-3 blocchi (index 0 to 7)
    #         for param in block.parameters():
    #             param.requires_grad = True
    #     else:
    #         for param in block.parameters():
    #             param.requires_grad = False
    # Assicurati che i parametri del classificatore rimangano sbloccati.

    # Rimuovi lo scheduler della Fase 1 se non lo vuoi continuare
    # scheduler = StepLR(optimizer, step_size=8, gamma=0.01) # Se vuoi un nuovo scheduler per questa fase
    # O, usa OneCycleLR che è ottimo per il fine-tuning
     # Numero di epoche per il fine-tuning (potrebbe essere di più)
     # Learning rate molto più basso per il fine-tuning

    # Re-inizializza l'ottimizzatore con il nuovo learning rate e i parametri aggiornati
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=fine_tune_lr, weight_decay=wd2)
    # Puoi anche usare OneCycleLR qui per un LR che varia dinamicamente
    if num_epochs_phase2>0:
        if scheduler2.lower()=='steplr':
            scheduler= StepLR(optim_backbone, step_size = num_steplr, gamma =0.1)
        elif scheduler2.lower()=='cosine':
            scheduler = CosineAnnealingLR(optim_backbone, T_max = num_epochs_phase1)
        elif scheduler2.lower()=='onecycle':
            scheduler = OneCycleLR(optimizer, max_lr=fine_tune_lr *gamma_onecycle , epochs=num_epochs_phase2, steps_per_epoch=len(dataloader))
        else:
            raise ValueError("learning rate scheduler not recognized")
        #scheduler = StepLR(optimizer,step_size= 3,gamma=0.1)
    stopper_phase2 = EarlyStopper(patience = 8)
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
        if scheduler2.lower() != 'onecycle':
            scheduler.step()
        dist.barrier()

        train_loss = running_loss / total_samples
        val_acc, val_precision, val_recall, val_loss = compute_scores(ddp_model, valloader_local, local_rank)
        ddp_model.train()
        val_loss_avg = logger.get_global_metric(val_loss)
        val_recall_avg = logger.get_global_metric(val_recall)
        val_precision_avg = logger.get_global_metric(val_precision)
        train_loss_avg = logger.get_global_metric(train_loss)

        
        if rank == 0:
            loss_history.append(train_loss_avg)
            accuracy_history.append(val_acc)
            precision_history.append(val_precision_avg)
            recall_history.append(val_recall_avg)
            loss_val_history.append(val_loss_avg)
            print(f"[Fase 2 - Epoch {epoch+1}/{num_epochs_phase2}] Train Loss: {train_loss_avg:.4f} Val Loss: {val_loss_avg:.4f} Val Acc: {val_acc:.4f} Prec: {val_precision_avg:.4f} Recall: {val_recall_avg:.4f} Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            with open(f"logs/log_{formatted_time}.txt",'a') as f:
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
        torch.save(ddp_model.module.state_dict(), f"logs/nofreeze_{formatted_time}.pth")
        with open(f"logs/log_{formatted_time}.txt",'a') as f:
            print("Model saved after training nicee",file=f)
        fig, axs= plt.subplots(1,3,figsize=(10,10))
        axs[0].plot(loss_history,'b')
        axs[0].plot(loss_val_history,'r')
        axs[0].legend(['train loss', 'val loss'])
        axs[0].set_title('Running  loss history')
        axs[1].plot(accuracy_history)
        axs[1].set_title('Accuracy history')
        axs[2].plot(precision_history,'-b')
        axs[2].plot(recall_history,'go')
        axs[2].legend(['precision', 'recall'])
        axs[2].set_title('Precision and recall')
        plt.savefig(f'logs/metrics_{formatted_time}.png')
    cleanup()
 
   

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    train()


