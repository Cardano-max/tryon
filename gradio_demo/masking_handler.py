import numpy as np
from PIL import Image
from masking_module.masking_module import generate_mask

def get_mask(image, category):
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Define task prompt and text prompt based on the category
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    if category == "upper_body":
        text_prompt = "Dress"
    elif category == "lower_body":
        text_prompt = "Dress"
    else:  # Full body
        text_prompt = "Dress"

    # Generate mask using the new masking module
    masks = generate_mask(
        image_input=Image.fromarray(image_np),
        image_url=None,
        task_prompt=task_prompt,
        text_prompt=text_prompt,
        dilate=10,
        merge_masks=True,
        return_rectangles=False,
        invert_mask=False
    )

    if masks and len(masks) > 0:
        # Return the first (and only, since merge_masks=True) mask
        return masks[0]
    else:
        # Return a blank mask if generation failed
        return np.zeros(image_np.shape[:2], dtype=np.uint8)

def test_masking():
    # Load a test image (replace with the path to your test image)
    test_image_path = "/Tes/3a757077-2880-401e-81f6-88890919f203.jpeg"
    test_image = Image.open(test_image_path)

    # Test with different categories
    categories = ["upper_body", "lower_body", "full_body"]

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
