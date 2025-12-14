#!/usr/bin/env python

from datasets import Dataset, DatasetDict
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ColorJitter, PILToTensor, RandomHorizontalFlip, RandomPerspective, RandomRotation, RandomVerticalFlip, Compose, RandomApply
from transformers import SegformerImageProcessor
os.environ["TRANSFORMERS_NO_TF"] = "1"
import albumentations as A
from dataset.dataset import LabelMap, load_image_mask_pairs
from torch import nn
import evaluate
import multiprocessing
import numpy as np 
import datetime

current_time = datetime.datetime.now()

formatted_time = current_time.strftime("%Y%m%d%H%M%S")

file_path = f'logs/seg_pret_logs/log_{formatted_time}.txt'

train_pipeline = A.Compose([
    A.Resize(height=320, width=320, p=1),      # Step 1: Size
    A.HorizontalFlip(p=0.5),                  # Step 2: Basic geometric
    #A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),  # Step 3: Dropout
    A.ConstrainedCoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.7, 0.7),
                        hole_width_range=(0.7, 0.7), mask_indices =[1], p=0.2),
    A.OneOf([
        A.ToGray(p=1.0),
        A.ChannelDropout(p=1.0)
    ], p=0.1),
    A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9),
                 angle_range=[0.1,0.5],
                 num_flare_circles_range=[2,8],
                 src_radius=100, src_color=(255, 255, 255),
                 p=0.4),
    A.SquareSymmetry(p=0.5),
    A.PlanckianJitter(p=0.3),
    A.RandomShadow(p=0.15),
    #A.Normalize(),                            # Step 7: Normalization
])


# Define paths
train_images = "dataset/solardk_segformer/segformer_google/gentofte_trainval/train/positive"
train_masks = "dataset/solardk_segformer/segformer_google/gentofte_trainval/train/mask"
val_images = "dataset/solardk_segformer/segformer_google/gentofte_trainval/val/positive"
val_masks = "dataset/solardk_segformer/segformer_google/gentofte_trainval/val/mask"

# Load datasets
train_ds = load_image_mask_pairs(train_images, train_masks)
val_ds = load_image_mask_pairs(val_images, val_masks)

# Combine into a DatasetDict
dataset = DatasetDict({
    "train": train_ds,
    "val": val_ds
})

from transformers import Trainer, SegformerForSemanticSegmentation, TrainingArguments
id2label = {0: 'background', 1: 'solar_panel'}
label2id = {'background': 0, 'solar_panel': 1}

processor = SegformerImageProcessor.from_pretrained("models/models--nvidia--mit-b2/snapshots/mit-b2", local_files_only=True, size = {"height":320,"width":320})

model = SegformerForSemanticSegmentation.from_pretrained(
    "models/models--nvidia--mit-b2/snapshots/mit-b2",
    local_files_only = True,
    id2label=id2label,
    label2id=label2id,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
state_dict = torch.load("logs/segclass_20250817132935.pth", map_location=device)
model.segformer.encoder.load_state_dict(state_dict, strict=False)

def preprocess_example(example, train = False):
    image = np.array(Image.open(example['image']).convert("RGB"))
    mask = np.array(Image.open(example['segmentation_mask']).convert("L"))
    if train:
        augmented = train_pipeline(image = image, mask = mask)
        image = augmented['image']
        mask = augmented['mask']
    else:
        image = image
        mask = mask

        # Apply LabelMap to the mask
    mask_tensor = torch.tensor(mask,dtype =torch.int32)
    labeled_mask = LabelMap(mask_tensor)

    # Use the processor to resize and normalize the image and mask
    inputs = processor(images=image, segmentation_maps=labeled_mask, return_tensors="pt")

    # Remove the batch dimension added by the processor
    inputs = {k: v.squeeze() for k, v in inputs.items()}

    return inputs

# Apply the preprocess function to the datasets
processed_train_ds = dataset["train"].map(lambda example: preprocess_example(example, train = True))
processed_val_ds = dataset["val"].map(lambda example: preprocess_example(example, train = False))

metric = evaluate.load("models/mean_iou")

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

from transformers import Trainer, SegformerForSemanticSegmentation, TrainingArguments

# Define training arguments
epochs = 40
lr = 0.0005
batch_size = 8 
weight_decay= 5e-5 

with open(file_path,'w') as f:
    f.write(f'Epochs: {epochs}\n lr: {lr}\n batch size: {batch_size}\n weight decay: {weight_decay}\n')

training_args = TrainingArguments(
    output_dir = f"logs/segformer_pret_{formatted_time}",
    learning_rate=lr,
    weight_decay = weight_decay,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="best",
   # eval_steps=30,
    logging_steps=50,
    lr_scheduler_type ='reduce_lr_on_plateau',
    disable_tqdm = True,
    metric_for_best_model = 'iou_solar_panel',
    greater_is_better = True,
    eval_accumulation_steps = 5,
    fp16 = True,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to = 'tensorboard',
    logging_dir =f'logs/seg_logs_pret_{formatted_time}',
    #hub_model_id=hub_model_id,
    #hub_strategy="end",
)


# Define the model
# Assuming id2label and label2id are defined in a previous cell and are available

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_ds, # Use the processed dataset
    eval_dataset=processed_val_ds,   # Use the processed dataset
    compute_metrics=compute_metrics,
)

trainer.train()



