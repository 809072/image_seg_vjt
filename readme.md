## Segmentation Mask Generation (COCO)

This script generates binary or multiclass segmentation masks from COCO-format datasets.

**Features:**
- Accepts image and annotation paths as input.
- Supports binary (foreground/background) or multiclass (per-category) masks.
- Masks are saved as `.npy` array which can be used for training a model for segmentation task and as `.png` for visualization.
- Skips images with no annotations or unreadable files.
- Handles overlapping masks in multiclass mode by assigning the last drawn object.
- Category IDs are mapped to contiguous indices (starting from 1).
- Reserved `0` as background in multiclass mode.
- Fully configurable via script.
- Can process number of images using `num_images` argument.
=======
# Semantic Segmentation Using DeepLabV3 on the COCO Dataset
This project focuses on semantic segmentation using DeepLabV3+ with a MiT-B2 encoder, trained on a processed subset of the COCO dataset. It includes everything from dataset preparation to model training and evaluation, all built using PyTorch Lightning for modularity and ease of use.

## 📦 Task 1: Dataset Preparation (COCO to Masks)
The script data_preprocessing.py transforms COCO annotations into segmentation masks.

## ✅ Key Features
Converts COCO-style annotations into pixel-wise multi-class masks.

Resolves overlaps using per-pixel max logic for accurate labeling.

Skips crowd-labeled or invalid annotation entries.

Processes up to 5,000 images at a time for efficiency.

Saves outputs in two folders:

RGB images to output/images/

Grayscale class masks to output/masks/

## ▶ Usage
bash
Copy
Edit
python data_preprocessing.py \
  --annotations path/to/instances_train2017.json \
  --images path/to/train2017 \
  --output path/to/output_dir \
  --max-images 5000
## 🧠 Task 2: Model Training (DeepLabV3 + MiT-B2)
Training is done using PyTorch Lightning and the segmentation_models_pytorch library with a DeepLabV3+ model and a MiT-B2 encoder.

## 🏗 Architecture Highlights
Utilizes DeepLabV3Plus from segmentation_models_pytorch.

Employs a MiT-B2 encoder pretrained on ImageNet.

Resizes images to 512×512 during training.

Uses Albumentations for image normalization and augmentation.

## ▶ Usage
bash
Copy
Edit
python train.py \
  --data_path path/to/output_dir \
  --epochs 50 \
  --lr 1e-4 \
  --num_classes 80 \
  --batch_size 16 \
  --img_size 512,512
## 🧪 Metrics
IoU (Jaccard Index) for each class and overall performance.

Dice Coefficient averaged across all classes.

Pixel Accuracy optionally logged via Weights & Biases (WandB).

## 🧰 Environment Setup
To ensure a clean and reproducible environment, this project uses uv for dependency management.

## ✅ Install dependencies
bash
Copy
Edit
pip install uv
uv venv
uv pip install -r requirements.txt
## 🗃 Project Structure
data_preprocessing.py — Converts COCO annotations into pixel-wise segmentation masks.

train.py — Training script with DeepLabV3+ and MiT-B2 using PyTorch Lightning.

trainning.ipynb — Notebook for exploring the dataset and model predictions.

requirements.txt — Dependency list used by uv for setting up the environment.

results/ — Contains sample input images, true masks, and predicted outputs.

input1.jpg — A sample image from the dataset.

mask1.png — The corresponding ground truth mask.

pred1.png — Model's predicted segmentation mask.

checkpoints/ — Stores trained model weights and checkpoints.

README.md — A detailed guide covering project overview, setup, and usage.

## ✨ Highlights
🔄 Complete Pipeline: From raw COCO annotations to a trained segmentation model.

🖼 Multi-class Segmentation: Generates pixel-level labeled masks for each object category.

💡 High-Performance Architecture: Uses DeepLabV3+ with MiT-B2 for strong results and fast inference.

⚙ Modular Training Setup: Built with PyTorch Lightning for easy customization and scaling.

🎯 Evaluation Metrics: Supports IoU, Dice Score, and optional Pixel Accuracy.

📊 WandB Logging: Real-time training and metric visualization via Weights & Biases.

🧪 Reproducibility: Fully isolated environment setup using uv.

📁 Clean Codebase: Easy to navigate and extend for other datasets or encoders.

🛠 Robust Preprocessing: Skips edge cases and ensures high-quality mask generation.

🔍 Visualization Ready: Provides sample outputs to quickly verify model performance.