import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

# Set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the checkpoint and model configuration
sam2_checkpoint = "segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "segment-anything-2/sam2_configs/sam2_hiera_l.yaml"

# Check if files exist
print(f"Checkpoint exists: {os.path.exists(sam2_checkpoint)}")
print(f"Config exists: {os.path.exists(model_cfg)}")

# Clear any previous Hydra config
GlobalHydra.instance().clear()

# Initialize Hydra
initialize(version_base=None, config_path="segment-anything-2/sam2_configs")

# Build SAM 2 model
sam2 = build_sam2(config_file="sam2_hiera_l.yaml", ckpt_path=sam2_checkpoint, device='cuda', apply_postprocessing=True)

# Initialize the automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(sam2)

def generate_mask(image_path, category):
    # Define the ontology mapping between the prompt and the class
    base_model = GroundedSAM2(
        ontology=CaptionOntology(
            {
                category: category
            }
        )
    )

    # Run inference on an image using the prompt
    results = base_model.predict(image_path)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Create a blank mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Fill the mask with white for the detected object
    if results.mask is not None:
        for single_mask in results.mask:  # Iterate over each detection's mask
            mask = np.logical_or(mask, single_mask).astype(np.uint8)

    # Convert to binary mask (0 and 255)
    binary_mask = mask * 255

    return image, binary_mask

def extract_cloth(image_path, category):
    # Define the ontology mapping between the prompt and the class
    base_model = GroundedSAM2(
        ontology=CaptionOntology(
            {
                category: category
            }
        )
    )

    # Run inference on an image using the prompt
    results = base_model.predict(image_path)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Create a white background
    cloth_extract = np.ones_like(image) * 255  # White background

    # Extract the cloth using the mask
    if results.mask is not None:
        for single_mask in results.mask:
            # Apply the mask to extract the cloth
            cloth_extract = np.where(single_mask[:,:,None], image, cloth_extract)

    return image, cloth_extract

def main():
    # Test mask generation
    image_path = "test_image.jpg"  # Replace with your test image path
    category = "Full Dress"
    
    image, mask = generate_mask(image_path, category)
    
    # Display the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(mask, cmap='gray')
    ax2.set_title(f'Binary Mask ({category})')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the binary mask
    cv2.imwrite('full_dress_mask.png', mask)

    # Test cloth extraction
    image, cloth_extract = extract_cloth(image_path, category)

    # Display the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(cloth_extract)
    ax2.set_title('Extracted Cloth')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the extracted cloth image
    cv2.imwrite('extracted_cloth.png', cv2.cvtColor(cloth_extract, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
