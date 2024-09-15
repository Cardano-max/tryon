import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Add the necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
tryon_dir = os.path.dirname(current_dir)
segment_anything_dir = os.path.join(tryon_dir, 'segment-anything-2')
sys.path.extend([tryon_dir, segment_anything_dir])

from sam2.build_sam import build_sam2
from sam2.predictor import SAM2ImagePredictor
from hydra import initialize, compose

# Initialize CUDA device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the paths for SAM model
SAM_CHECKPOINT_PATH = os.path.join(segment_anything_dir, "checkpoints", "sam2_hiera_large.pt")
SAM_CONFIG_PATH = os.path.join(segment_anything_dir, "sam2_configs", "sam2_hiera_l.yaml")

# Initialize Hydra
initialize(version_base=None, config_path=os.path.join(segment_anything_dir, "sam2_configs"))

# Build SAM 2 model
sam2 = build_sam2(config_file="sam2_hiera_l.yaml", ckpt_path=SAM_CHECKPOINT_PATH, device=DEVICE, apply_postprocessing=True)

# Initialize the SAM2 predictor
predictor = SAM2ImagePredictor(sam2)

def generate_mask(
    image_input: Image.Image,
    category: str,
    dilate: int = 10,
    invert_mask: bool = False
) -> np.ndarray:
    # Convert PIL Image to numpy array
    image_np = np.array(image_input.convert("RGB"))

    # Set the image for the predictor
    predictor.set_image(image_np)

    # Generate masks for the entire image
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=None,
        multimask_output=True
    )

    # Combine all masks
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    # Apply dilation if specified
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Invert mask if specified
    if invert_mask:
        combined_mask = cv2.bitwise_not(combined_mask)

    return combined_mask

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