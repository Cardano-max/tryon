import os
import torch
import numpy as np
from PIL import Image
import cv2
import supervision as sv
from hydra import initialize, compose
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
tryon_dir = os.path.dirname(current_dir)
segment_anything_dir = os.path.join(tryon_dir, 'segment-anything-2')
sam2_checkpoint = os.path.join(segment_anything_dir, "checkpoints", "sam2_hiera_large.pt")
model_cfg = os.path.join(segment_anything_dir, "sam2_configs", "sam2_hiera_l.yaml")

# Initialize CUDA device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Hydra
initialize(version_base=None, config_path=os.path.join(segment_anything_dir, "sam2_configs"))

# Build SAM 2 model
sam2 = build_sam2(config_file="sam2_hiera_l.yaml", ckpt_path=sam2_checkpoint, device=DEVICE, apply_postprocessing=True)

# Initialize the automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# Monkey patch the build_sam2 function and load_SAM function in autodistill_grounded_sam_2
from autodistill_grounded_sam_2 import helpers
def patched_build_sam2(model_cfg, checkpoint, device=DEVICE):
    return build_sam2(config_file=model_cfg, ckpt_path=checkpoint, device=device, apply_postprocessing=True)

def patched_load_SAM():
    model_cfg = os.path.join(segment_anything_dir, "sam2_configs", "sam2_hiera_l.yaml")
    checkpoint = os.path.join(segment_anything_dir, "checkpoints", "sam2_hiera_large.pt")
    sam2 = patched_build_sam2(model_cfg, checkpoint, DEVICE)
    return sam2

helpers.build_sam2 = patched_build_sam2
helpers.load_SAM = patched_load_SAM

def generate_mask(image_path, category):
    # Load the image
    image = np.array(Image.open(image_path).convert("RGB"))

    # Generate masks using SAM2
    masks = mask_generator.generate(image)

    # Use GroundedSAM2 for specific category detection
    base_model = GroundedSAM2(
        ontology=CaptionOntology(
            {
                category: category
            }
        )
    )

    # Run inference on the image
    results = base_model.predict(image_path)

    # Create a binary mask
    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if results.mask is not None:
        for single_mask in results.mask:
            binary_mask = np.logical_or(binary_mask, single_mask).astype(np.uint8)

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
