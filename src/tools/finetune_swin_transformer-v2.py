# ViT： https://huggingface.co/docs/transformers/v4.39.2/ja/tasks/image_classification
# Create database： https://huggingface.co/docs/datasets/create_dataset
# use of wandb：https://docs.wandb.ai/ja/guides/integrations/huggingface/
# Reference blog： https://zenn.dev/platina/scraps/f98377590b3249

#####################################################################################
# Please login huggingface with the following command before running this script:
#   $ transformers-cli login
# For huggingface access token, please get it from https://huggingface.co/settings/tokens
# You may need to input wandb API key. Please get it from https://wandb.ai/authorize
#####################################################################################

# 本コードはGPUを使用してください（CPUには対応していません）

from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import pipeline
import evaluate
import numpy as np
import os
import shutil
from system_setting import setup_env_data

# set the system settings
setup_env_data()
MODEL_FOLDER = os.environ.get('MODEL_FOLDER')
ORIGINAL_MODEL = MODEL_FOLDER + "swinv2"
OUTPUT_MODEL = MODEL_FOLDER + "swinv2_finetuning"

DATA_FOLDER = os.environ.get('DATA_FOLDER')
SAMPLE_DATASET = DATA_FOLDER + "time_series_data/images"

# Define the data processing function
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

# Define the compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Load the dataset
dataset = load_dataset("imagefolder", data_dir=SAMPLE_DATASET)
print(dataset["train"][0])

# Define the label2id and id2label
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# load the image processor
image_processor = AutoImageProcessor.from_pretrained(ORIGINAL_MODEL, use_fast=True)

# Define the transforms
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

# apply the data transform function to the dataset
dataset = dataset.with_transform(transforms)

# Define the data collator
data_collator = DefaultDataCollator()

# Load the accuracy metric
accuracy = evaluate.load("accuracy")

# Load the model
model = AutoModelForImageClassification.from_pretrained(
    ORIGINAL_MODEL,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL,
    remove_unused_columns=False,
    eval_strategy="epoch",   # save_strategy="epoch"も設定する必要あり
    #eval_strategy="no",     
    save_strategy="epoch",
    #save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=30,     # エポックの設定が重要（大きくする）
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    #report_to="wandb",    # wandbを使う
    #report_to="tensorboard",    # tensorboardを使う
    report_to="none",      # wandbを使わない
    push_to_hub=False,     # don't upload it to HugginfFace
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
# trainer.evaluate()   # Tensorboardを使うか、標準出力のためにはCallable関数の設定が必要

# Save the model
trainer.save_model(OUTPUT_MODEL)

# Copy preprocessor_config.json to the output folder
shutil.copyfile(ORIGINAL_MODEL + "/preprocessor_config.json", OUTPUT_MODEL + "/preprocessor_config.json")

# Prepare dataset for nference
dataset = load_dataset("imagefolder", data_dir=SAMPLE_DATASET, split="test")

# Evaluate the model with validation dataset
hit_cnt = 0
cat_hit_cnt = {}
for idx in range(0, len(dataset["label"])):
    print("[" + str(idx) + "]\nGiven class: " + id2label[str(dataset["label"][idx])])
    image = dataset["image"][idx]

    # Classify the image data
    classifier = pipeline("image-classification", model=OUTPUT_MODEL, device=0)  # 0:GPU
    res = classifier(image)

    print("Predicted class:")
    ranking = 0
    for result in res:
        if result["label"] == id2label[str(dataset["label"][idx])]:
            info = " Matched!"
            if ranking == 0:
                hit_cnt += 1
                if result["label"] in cat_hit_cnt:
                    cat_hit_cnt[result["label"]] += 1
                else:
                    cat_hit_cnt[result["label"]] = 1
        else:
            info = ""
        print(" " + str(ranking+1) + ": " + result["label"] + " [" + str(result["score"]) + "] " + info)
        ranking += 1
    print()

print("Accuracy: " + str((hit_cnt / len(dataset["label"]) * 100)) + " %\n")

for key, value in cat_hit_cnt.items():
    print(key + ": " + str(value))
