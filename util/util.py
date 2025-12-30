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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from itertools import chain, product
import argparse
from . import config

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
        precision = precision_score(all_labels, all_preds, pos_label = 0,average='binary')
        recall = recall_score(all_labels, all_preds, pos_label = 0, average='binary')
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
        precision = precision_score(all_labels, all_preds, pos_label=0, average='binary')
        recall = recall_score(all_labels, all_preds, pos_label=0, average='binary')
    return accuracy, precision, recall, val_loss


def output_label(label):  # Define a function to map numeric labels to class names
    output_mapping = {  # Dictionary mapping numeric labels to class names
        0: "solar",
        1: "no solar",
         }
    input = (label.item() if type(label) == torch.Tensor else label)  # Convert tensor to scalar if needed
    return output_mapping[input]  # Return the corresponding class name

### `create_confusion_matrix` Function

def create_confusion_matrix(model, loader, device, file_path, fig_path, seg = False):  # Define a function to create a confusion matrix
    class_correct = [0. for _ in range(2)]  # List to store correctly predicted counts for each class
    total_correct = [0. for _ in range(2)]  # List to store total counts for each class

    predictions_list = []  # List to store all predictions
    labels_list = []  # List to store all true labels

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for data, target in loader:  # Iterate over batches of data and targets
            data, target = data.to(device), target.to(device)  # Move data and target to the specified device
            if seg:
                outputs =model(pixel_values = data)
                outputs = outputs.logits
            else:
                outputs = model(data)  # Get model predictions
            predicted = torch.max(outputs, 1)[1]  # Get the index of the max probability
            c = (predicted == target ) # Check if predictions are correct
            labels_list.append(target)  # Append true labels to the list
            predictions_list.append(predicted)  # Append predictions to the list

            for i in range(len(target)):  # Iterate over each label in the batch
                label = target[i]  # Get the true label
                class_correct[label] += c[i].item()  # Increment correct count for the label if prediction is correct
                total_correct[label] += 1  # Increment total count for the label

    for i in range(2):  # Iterate over all classes
        with open(file_path, 'a') as f:
            f.write("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))  # Print accuracy for each class

    predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]  # Flatten predictions list
    labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]  # Flatten labels list
    predictions_l = list(chain.from_iterable(predictions_l))  # Convert list of lists to a single list
    labels_l = list(chain.from_iterable(labels_l))  # Convert list of lists to a single list
    with open(file_path,'a') as f:
        f.write("Classification report for EfficientNet :\n%s\n" % (classification_report(labels_l, predictions_l)))  # Print classification report

    cm = confusion_matrix(labels_l, predictions_l)  # Compute confusion matrix

    classes = ['solar', 'no_solar']  # Class names
    plt.figure(figsize=(8, 8))  # Create a new figure
    plt.imshow(cm, cmap=plt.cm.Reds)  # Display the confusion matrix
    plt.title(' Confusion Matrix ')  # Title for the plot
    plt.colorbar()  # Add a color bar
    plt.xticks(np.arange(2), classes, rotation=90)  # Set x-axis ticks and labels
    plt.yticks(np.arange(2), classes)  # Set y-axis ticks and labels

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):  # Iterate over confusion matrix elements
        plt.text(j, i, "{:0.2f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > 500 else "black")  # Annotate each cell
    
    plt.tight_layout()  # Adjust layout so labels fit nicely
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # Save to file
    plt.close()
    return



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--config', type=str, default='configs/danish_with_bbr_and_google/danish_with_bbr_and_google_efficientnet_v2s.yaml', help='config file')
    parser.add_argument('opts', help='see configs/danish_with_bbr_and_google/danish_with_bbr_and_google_efficientnet_v2s.yaml', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def count_activation(root, cls=nn.SiLU):
    return sum(1 for m in root.modules() if isinstance(m, cls))

def replace_activation(root, old_cls=nn.SiLU, new_cls=nn.ReLU):
    for name, child in list(root._modules.items()):
        if isinstance(child, old_cls):
            root._modules[name] = new_cls()
        else:
            # recurse into submodules
            replace_activation(child, old_cls, new_cls)


def get_warmup_then_steplr(optimizer, num_warmup_steps, step_size, gamma=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup: scala linearmente da 0 a 1
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Dopo warmup: applica step decay
            steps_after_warmup = current_step - num_warmup_steps
            return gamma ** (steps_after_warmup // step_size)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
