import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

# Add the segment-anything-2 directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
tryon_dir = os.path.dirname(current_dir)
segment_anything_dir = os.path.join(tryon_dir, 'segment-anything-2')
sys.path.append(segment_anything_dir)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Set up paths
sam2_checkpoint = os.path.join(segment_anything_dir, "checkpoints", "sam2_hiera_large.pt")
model_cfg = os.path.join(segment_anything_dir, "sam2_configs", "sam2_hiera_l.yaml")

# Initialize CUDA device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build SAM 2 model
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)

# Initialize the automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100
)

def generate_mask(image_path, category):
    # Load the image
    image = np.array(Image.open(image_path).convert("RGB"))

    # Generate masks using SAM2
    masks = mask_generator.generate(image)

    # Create a binary mask (combining all masks)
    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        binary_mask = np.logical_or(binary_mask, mask['segmentation']).astype(np.uint8)

    # Convert to binary mask (0 and 255)
    binary_mask = binary_mask * 255

    return binary_mask

if __name__ == "__main__":
    # Test the mask generation
    test_image_path = "/imagePIL.png"
    category = "Full Dress"
    mask = generate_mask(test_image_path, category)

    # Save the generated mask
    cv2.imwrite('test_mask.png', mask)
    print("Mask generated and saved as test_mask.png")
