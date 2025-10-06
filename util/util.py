import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as dist
from sklearn.metrics import precision_score, recall_score, accuracy_score, jaccard_score
import torch.nn as nn
import shutil
import os


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

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=-2,
            reduce_labels= False,
        )

    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics

def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean


def copy_positive(model, dataloader, classes, device, output_dir):
     model.to(device)
     model.eval()
     for images, filenames in dataloader:
         filenames = np.array(filenames)
         images = images.to(device)
         outputs = model(images)
         preds = torch.argmax(outputs, dim=1)
         output_class=np.array([classes[int(p)] for p in preds])
         positives = filenames[output_class=='positive']
         for file in positives:
            shutil.copy(os.path.join(dataloader.dataset.img_dir, file), output_dir)
     print(f"Copied everything to {output_dir}")

def init_write(file_path, batch_size, epochs_1, epochs_2, learning_rate_2,  wd1, wd2, scheduler1, scheduler2, scheduler_diff,  num_unfrozen1, num_unfrozen2, num_steplr,gamma_onecycle, learning_diff, learning_rate1= None):
    with open(file_path,'w') as f:
        f.write('CONFIGS:\n')
    with open(file_path, 'a') as f:
        f.write(f'BATCH SIZE: {batch_size}\n EPOCHS PHASE 1: {epochs_1} \n EPOCHS PHASE 2: {epochs_2} \n LEARNING RATE CLASSIFIER PHASE1: {learning_diff}\n')
        if learning_rate1:
            f.write(f'LEARNING RATE BACKBONE PHASE1: {learning_rate1}\n')
        f.write(f'LEARNING RATE PHASE2: {learning_rate_2}\n WEIGHT DECAY PHASE 1: {wd1}\n WEIGHT DECAY PHASE2: {wd2}\n SCHEDULER PHASE1 :{scheduler1}\n SCHEDULER PHASE2: {scheduler2} SCHEDULER DIFF: {scheduler_diff}\n NUMBER OF UNFROZEN LAYERS PHASE1: {num_unfrozen1} \n NUMBER OF UNFROZEN LAYERS PHASE2: {num_unfrozen2}\n NUMBER OF EPOCHS BEFORE REDUCING LR WITH STEPLR:{num_steplr}\n GAMMA ONECYCLE:{gamma_onecycle}\n')

def compute_scores_seg(model, dataloader, device):
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
            outputs = model(pixel_values = inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            _, preds = torch.max(logits, 1)
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

