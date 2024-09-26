import os
from FlorenceMasking import FlorenceMasking
from PIL import Image
import matplotlib.pyplot as plt

def test_florence_masking():
    # Initialize FlorenceMasking
    florence_masking = FlorenceMasking()

    # Path to the test image
    image_path = "/tryon/masking_module2/ArbitryonMasking/Florence/hania.jpg"

    # Ensure the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return

    # Test the masking
    mask = florence_masking.get_mask(image_path)

    if mask is not None:
        # Load the original image
        original_image = Image.open(image_path)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Display the original image
        ax1.imshow(original_image)
        ax1.set_title("Original Image")
        ax1.axis('off')

        # Display the mask
        ax2.imshow(mask, cmap='gray')
        ax2.set_title("Generated Mask")
        ax2.axis('off')

        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()

        # Save the figure
        output_path = "/tryon/masking_module2/ArbitryonMasking/Florence/masking_result.png"
        plt.savefig(output_path)
        print(f"Masking result saved to {output_path}")
    else:
        print("Failed to generate mask")

if __name__ == "__main__":
    test_florence_masking()
