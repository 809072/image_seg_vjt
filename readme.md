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
