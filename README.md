# FLUX.1-Fill-dev LoRA Training

This repository provides code for training LoRAs (Low-Rank Adaptations) for the [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) model. It is based on the repository [Sebastian-Zok/FLUX-Fill-LoRa-Training](https://github.com/Sebastian-Zok/FLUX-Fill-LoRa-Training) but has been modified to fit my use case.

## Key Features

* **Customizable Data Directories**: You can specify separate directories for instance, class, and validation data.
* **Data Organization**: Each data directory can contain the following files, which are grouped by `<name>`:
    * `<name>.png`: The image file to be used for training.
    * `<name>_mask.png`: A mask image file where the area to be filled is painted white.
    * `<name>.txt`: A text file containing the image caption.

## Example

See the `tutorial` directory for an example of how to organize your data.

## Usage

```bash
git clone https://github.com/hvppycoding/hvppyfluxfill.git
cd hvppyfluxfill
pip install -e .

accelerate config default

huggingface-cli login
# Enter huggingface API token

pip install wandb
```

```bash
cd tutorial
accelerate launch ../hvppyfluxfill_main.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-Fill-dev" \
  --instance_prompt="A sks dog" \
  --instance_data_dir="./train_data" \
  --with_prior_preservation \
  --class_prompt="A dog" \
  --class_data_dir="./class_data" \
  --validation_prompt="A sks dog" \
  --validation_data_dir="./validation_data" \
  --validation_epochs=25 \
  --validation_repeats=2 \
  --output_dir="sks-dog-flux-fill-training" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=1 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=200 \
  --max_train_steps=1000 \
  --rank=8 \
  --checkpointing_steps=100 \
  --seed="0" \
  --resume_from_checkpoint=latest \
  --push_to_hub
```
