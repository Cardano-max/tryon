import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import requests
from typing import Optional, List
import time
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from hydra import initialize, compose

# Initialize CUDA device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the paths for SAM model
SAM_CHECKPOINT_PATH = "checkpoints/sam2_hiera_large.pt"
SAM_CONFIG_PATH = "sam2_configs/sam2_hiera_l.yaml"

# Initialize Hydra
initialize(version_base=None, config_path="sam2_configs")

# Build SAM 2 model
sam2 = build_sam2(config_file="sam2_hiera_l.yaml", ckpt_path=SAM_CHECKPOINT_PATH, device=DEVICE, apply_postprocessing=True)

# Initialize the automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# Initialize GroundedSAM2
base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            "Full Dress": "Full Dress",
            "Upper Body": "Upper Body",
            "Lower Body": "Lower Body"
        }
    )
)

def generate_mask(
    image_input: Image.Image,
    category: str,
    dilate: int = 10,
    invert_mask: bool = False
) -> np.ndarray:
    # Convert PIL Image to numpy array
    image_np = np.array(image_input.convert("RGB"))

    # Run inference using GroundedSAM2
    results = base_model.predict(image_np, ontology={category: category})

    # Create a blank mask
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

    # Fill the mask with white for the detected object
    if results.mask is not None:
        for single_mask in results.mask:
            mask = np.logical_or(mask, single_mask).astype(np.uint8)

    # Convert to binary mask (0 and 255)
    binary_mask = mask * 255

    # Apply dilation if specified
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Invert mask if specified
    if invert_mask:
        binary_mask = cv2.bitwise_not(binary_mask)

    return binary_mask

# Test function
def test_masking():
    # Load a test image
    test_image_path = "/imagePIL.png"
    test_image = Image.open(test_image_path)

    # Test with different categories
    categories = ["Full Dress", "Upper Body", "Lower Body"]

    for category in categories:
        print(f"Testing masking for {category}...")
        mask = generate_mask(test_image, category)

        # Save the generated mask
        mask_image = Image.fromarray(mask)
        mask_image.save(f"test_mask_{category.lower().replace(' ', '_')}.png")
        print(f"Mask for {category} saved as test_mask_{category.lower().replace(' ', '_')}.png")

    print("Masking test completed.")

if __name__ == "__main__":
    test_masking()