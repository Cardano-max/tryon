import numpy as np
from PIL import Image
from masking_module.masking import generate_mask

def get_mask(image, category):
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Define task prompt and text prompt based on the category
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    if category == "upper_body":
        text_prompt = "A person wearing an upper body garment"
    elif category == "lower_body":
        text_prompt = "A person wearing a lower body garment"
    else:  # Full body
        text_prompt = "A person wearing a full body garment"

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
