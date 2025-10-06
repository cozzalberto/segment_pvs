
#!/usr/bin/env python

from datasets import Dataset, DatasetDict
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import Trainer, SegformerForSemanticSegmentation, TrainingArguments, SegformerImageProcessor
import albumentations as A
from dataset.dataset import LabelMap, load_image_mask_pairs
from torch import nn
import evaluate
import multiprocessing
import numpy as np
import datetime


test_images =  "dataset/solardk_segformer/segformer_google/test_bologna/positive/"

test_masks = "dataset/solardk_segformer/segformer_google/test_bologna/mask/"

# Load datasets
test_ds = load_image_mask_pairs(test_images, test_masks)

current_time = datetime.datetime.now()

formatted_time = current_time.strftime("%Y%m%d%H%M%S")

file_path = f'logs/test/seg_logs/log_{formatted_time}.txt'

# Combine into a DatasetDict
dataset = DatasetDict({
    "test": test_ds,
})

id2label = {0: 'background', 1: 'solar_panel'}
label2id = {'background': 0, 'solar_panel': 1}


processor = SegformerImageProcessor.from_pretrained("models/models--nvidia--mit-b2/snapshots/mit-b2", size ={'height': 320,'width':320}, local_files_only=True)  # o tuo config.json custom
# Carica il modello da checkpoint
model = SegformerForSemanticSegmentation.from_pretrained(
    "logs/segformer_finetuning2_20250813105537/checkpoint-6752",  # <-- cambia con il percorso giusto
    local_files_only=True,
    id2label = id2label,
    label2id = label2id
)
model.eval()

def preprocess_example(example, train = False):
    image = np.array(Image.open(example['image']).convert("RGB"))
    mask = np.array(Image.open(example['segmentation_mask']).convert("L"))
        # Apply LabelMap to the mask
    mask_tensor = torch.tensor(mask,dtype =torch.int32)
    labeled_mask = LabelMap(mask_tensor)

    # Use the processor to resize and normalize the image and mask
    inputs = processor(images=image, segmentation_maps=labeled_mask, return_tensors="pt")

    # Remove the batch dimension added by the processor
    inputs = {k: v.squeeze() for k, v in inputs.items()}

    return inputs

# Apply the preprocess function to the datasets
processed_test_ds = dataset["test"].map(lambda example: preprocess_example(example, train = False))
print(processed_test_ds)
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

training_args = TrainingArguments(
    output_dir="logs/results_segtest",   # obbligatorio, serve come cartella di lavoro
    per_device_eval_batch_size=8,
    do_train=False,
    do_eval=True,
    logging_dir="logs/test_seglogs",     # opzionale
    report_to="none",
    remove_unused_columns=False,   # <--- aggiungi questo# evita logging su wandb/hub se non ti serve
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=processed_test_ds,
    compute_metrics=compute_metrics,  # optional
)

predictions = trainer.evaluate()
import pprint
with open(file_path,'w') as f:
    pprint.pprint(predictions, stream=f)
