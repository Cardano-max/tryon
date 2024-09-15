import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from masking_module.masking_module import generate_mask

def get_mask(image, category):
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Map category to the format expected by the masking module
    category_mapping = {
        "upper_body": "Upper Body",
        "lower_body": "Lower Body",
        "dresses": "Full Dress"
    }
    mapped_category = category_mapping.get(category, "Full Dress")

    # Generate mask using the new masking module
    mask = generate_mask(Image.fromarray(image_np), mapped_category)

    return mask

def test_masking():
    # Load the test image
    test_image_path = "/imagePIL.png"
    test_image = Image.open(test_image_path)

    # Test with different categories
    categories = ["upper_body", "lower_body", "dresses"]

    for category in categories:
        print(f"Testing masking for {category}...")
        mask = get_mask(test_image, category)

        # Save the generated mask
        mask_image = Image.fromarray(mask)
        mask_image.save(f"test_mask_{category}.png")
        print(f"Mask for {category} saved as test_mask_{category}.png")

    print("Masking test completed.")

if __name__ == "__main__":
    test_masking()
