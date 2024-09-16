from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the ontology mapping between the prompt and the class
base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            "Full Dress": "Full Dress"
        }
    )
)

# Path to the image
# image_path = "/content/00011_00.png"
image_path = "/tryon/masking_module/imagePIL.png"

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

# Optionally, save the binary mask
cv2.imwrite('/tryon/masking_module/output_mask.png', binary_mask)