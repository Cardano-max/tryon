# -- coding: utf-8 --
"""
Masking implementation using GroundedSAM2
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

def run_masking(image_path, output_path):
    # Define the ontology mapping between the prompt and the class
    base_model = GroundedSAM2(
        ontology=CaptionOntology(
            {
                "Full Dress": "Full Dress"
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
        # The mask attribute is a 3D array where the first two dimensions are the image dimensions
        # and the third dimension is the number of detections
        for single_mask in results.mask:  # Iterate over each detection's mask
            mask = np.logical_or(mask, single_mask).astype(np.uint8)

    # Convert to binary mask (0 and 255)
    binary_mask = mask * 255

    # Display the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(binary_mask, cmap='gray')
    ax2.set_title('Binary Mask (Full Dress)')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

    # Save the binary mask
    cv2.imwrite(output_path, binary_mask)

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    output_path = "path/to/output/full_dress_mask.png"
    run_masking(image_path, output_path)
