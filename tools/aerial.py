#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
import torch
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as dist
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datetime import timedelta
from torchsummary import summary
import sklearn
from torchvision.models import efficientnet_b0

class CustomDataset(Dataset):
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
        image = Image.open(img_path).convert('RGB') # Ensure image is in RGB

        if self.transform:
            image = self.transform(image)

        return image, label

class EarlyStopper():   

    def __init__(self, patience = 5):
        """
        Initializes the EarlyStopper with a specified patience level.

        Parameters:
        - patience (int): Number of epochs with no improvement after which training will be stopped.
        """
        self.patience = patience
        self.best_loss = np.inf
        self.epochs_no_improvement = 0


    def early_stopping(self, validation_loss):
        """
        Determines if early stopping should be triggered based on validation loss.

        Parameters:
        - patience (int): Number of epochs with no improvement after which training will be stopped.
        - validation_loss (float): Current validation loss.
        - best_loss (float): Best validation loss observed so far.
        - epochs_no_improvement (int): Number of epochs since the last improvement.

        Returns:
        - bool: True if early stopping should be triggered, False otherwise.
        """
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.epochs_no_improvement = 0
        else:
            self.epochs_no_improvement += 1

        return self.epochs_no_improvement >= self.patience, self.best_loss, self.epochs_no_improvement

class MetricLogger():
    def __init__(self, device, world_size):
        self.device = device
        self.world_size = world_size

    def get_global_metric(self, metric):
        metric_tensor = torch.tensor(metric, device=self.device)
        torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
        metric_avg = metric_tensor.item() / self.world_size
        return metric_avg

def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120), rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

from sklearn.metrics import precision_score, recall_score, accuracy_score
def compute_scores(model, dataloader, device):
    # Collect all predictions and true labels
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        # Compute precision and recall
        val_loss= total_loss/total_samples
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
    return accuracy, precision, recall, val_loss

# Define transformations (you can use your existing transformations)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.CenterCrop(384),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
])
training_transform = transforms.Compose([
                                        transforms.Resize((384, 384)),
                                        transforms.CenterCrop(384),
                                        transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                        transforms.RandomRotation(30),     #Rotates the image to a specified angel
                                        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                        transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #Normalize all the images
                                        ])

# Create the dataset
batch_size=4

trainset_filtered = CustomDataset(root_dir='./danish_dataset2/solardk_dataset_neurips_v2/gentofte_trainval/train', classes=['positive', 'negative'], transform=training_transform)

# You can then create a DataLoader from this dataset
#trainloader = torch.utils.data.DataLoader(trainset_filtered, batch_size=batch_size, shuffle=True, num_workers=4)

valset = CustomDataset(root_dir='./danish_dataset2/solardk_dataset_neurips_v2/gentofte_trainval/val', classes=['positive', 'negative'], transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
        

def train():
    rank, world_size, local_rank= setup()
    torch.cuda.set_device(local_rank)
    checkpoint = torch.load("dfc15_pretrained_imagenet1K_efficientnetb0.pth.tar", map_location=f"cuda:{local_rank}")
# Istanzia il modello
    model = efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features,8)
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Carica i pesi
    model.load_state_dict(checkpoint['state_dict'])  # o solo checkpoint se è un semplice dict
    model.eval()
    model.avgpool = nn.AdaptiveAvgPool2d((2,2))
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(num_ftrs*2*2,512),
                        nn.ReLU(),
                        nn.Linear(512,256),
                        nn.ReLU(),
                        nn.Dropout(0.4),
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
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    stopper=EarlyStopper(patience=5)
    #scheduler = OneCycleLR(optimizer, 0.001, total_steps=None, epochs=num_epochs, steps_per_epoch=len(dataloader))
    #scheduler = StepLR(optimizer,step_size=10,gamma=0.1)
    logger = MetricLogger(local_rank, world_size)
    
    for param in ddp_model.module.parameters():
        param.requires_grad = False

    # Assicurati che i parametri del classificatore siano sbloccati
    for param in ddp_model.module.classifier.parameters():
        param.requires_grad = True
    for param in ddp_model.module.features[-2:].parameters():
        param.requires_grad = True

    backbone_params = filter(lambda p: p.requires_grad, ddp_model.module.features.parameters())
    classifier_params = filter(lambda p: p.requires_grad, ddp_model.module.classifier.parameters())
    # Definisci l'ottimizzatore per i parametri che richiedono il gradiente (solo il classificatore inizialmente)
    optimizer = optim.Adam([{'params':backbone_params,'lr':5e-4},{'params': classifier_params, 'lr':0.003}], weight_decay=5e-4) # LR più basso, aggiungi weight_decay
    #scheduler = StepLR(optimizer,step_size=11,gamma=0.1)
    #Definisci l'ottimizzatore per i parametri che richiedono il gradiente (solo il classificatore inizialmente)   
    num_epochs_phase1 = 10 # Addestra solo il classificatore per 5 epoche (puoi sperimentare)
    if rank == 0:
        print("--- Fase 1: Addestramento del Classificatore ---")

    for epoch in range(num_epochs_phase1):
        sampler.set_epoch(epoch)
        ddp_model.train() # Imposta il modello in modalità training
        running_loss = 0.0
        total_samples = 0
        stop_flag = torch.tensor(0, device = local_rank)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(local_rank), labels.to(local_rank)
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        #scheduler.step()  
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
            
            should_stop, best_loss, epochs_no_improvement = stopper.early_stopping(val_loss_avg)
            if should_stop:
                print(f'early stopping in phase1 at epoch {epoch+1}\n')
                stop_flag +=1 
                stop_flag = torch.tensor(stop_flag, device = local_rank)
        torch.distributed.broadcast(stop_flag, src = 0)
        if stop_flag.item()>0:
            break



    # --- FASE 2: Fine-tuning dell'intera rete (o di strati selezionati) ---
    # Scongela tutti i parametri (o solo gli ultimi blocchi, vedi note sotto)
    for param in ddp_model.module.features[-3:].parameters():
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
    num_epochs_phase2 = 20 # Numero di epoche per il fine-tuning (potrebbe essere di più)
    fine_tune_lr = 5e-5 # Learning rate molto più basso per il fine-tuning

    # Re-inizializza l'ottimizzatore con il nuovo learning rate e i parametri aggiornati
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=fine_tune_lr, weight_decay=5e-4)
    # Puoi anche usare OneCycleLR qui per un LR che varia dinamicamente
    scheduler = OneCycleLR(optimizer, max_lr=fine_tune_lr * 20, epochs=num_epochs_phase2, steps_per_epoch=len(dataloader))
    #scheduler = StepLR(optimizer,step_size= 3,gamma=0.1)
    stopper_phase2 = EarlyStopper(patience = 5)
    if rank == 0:
        print("\n--- Fase 2: Fine-tuning dell'intera rete (o sezioni) ---")

    for epoch in range(num_epochs_phase2):
        sampler.set_epoch(epoch + num_epochs_phase1) # Continua con l'indice di epoca corretto
        ddp_model.train()
        running_loss = 0.0
        total_samples = 0
        stop_flag = torch.tensor(0, device = local_rank)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(local_rank), labels.to(local_rank)
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step() # Step del scheduler dopo ogni batch
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        #scheduler.step()
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
            should_stop, best_loss, epochs_no_improvement = stopper_phase2.early_stopping(val_loss_avg)
            if should_stop:
                print(f'early stopping in phase2 at epoch {epoch+1}\n')
                stop_flag +=1
                stop_flag = torch.tensor(stop_flag, device = local_rank)
        torch.distributed.broadcast(stop_flag, src = 0)
        if stop_flag.item()>0:
            print('ciao')
            break
  

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "nofreeze_aug_partial.pth")
        with open('Efficient_Net/outputs.txt','a') as f:
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
        plt.savefig('Efficient_Net/metrics_aug.png')
    cleanup()
 
   

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    train()


