import sys
import os
import time
import traceback
import numpy as np
import torch
from PIL import Image
import cv2
from queue import Queue
from threading import Lock, Event
import modules1.config
import modules1.async_worker as worker
import modules1.constants as constants
import modules1.flags as flags
from modules1.util import HWC3, resize_image
import os
print(sys.path)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from preprocess.masking import Masking

# Initialize the Masking class
masking = Masking()

# Use it in your code
def generate_mask(person_image, category="dresses"):
    if not isinstance(person_image, Image.Image):
        person_image = Image.fromarray(person_image)

    print("Generating mask...")
    try:
        inpaint_mask = masking.get_mask(person_image, category=category)
        print("Mask generated successfully.")
    except Exception as e:
        print(f"Error occurred while generating mask: {str(e)}")
        raise e
    return np.array(inpaint_mask)  # Convert to numpy array


def virtual_try_on(person_image_path, prompt, category="dresses", output_path=None):
    try:
        # Load person image
        print(f"Loading person image from: {person_image_path}")
        try:
            person_image = Image.open(person_image_path)
            person_image = np.array(person_image)
            print("Person image loaded successfully.")
        except Exception as e:
            print(f"Error occurred while loading person image: {str(e)}")
            raise e

        # Generate mask
        print("Generating mask...")
        try:
            inpaint_mask = generate_mask(person_image, category)
            print("Mask generated successfully.")
        except Exception as e:
            print(f"Error occurred while generating mask: {str(e)}")
            raise e

        # Get the original dimensions of the person image
        orig_person_h, orig_person_w = person_image.shape[:2]

        # Calculate the aspect ratio of the person image
        person_aspect_ratio = orig_person_h / orig_person_w

        # Set target width and calculate corresponding height to maintain aspect ratio
        target_width = 1024
        target_height = int(target_width * person_aspect_ratio)

        # Ensure target height is also 1024 at maximum
        if target_height > 1024:
            target_height = 1024
            target_width = int(target_height / person_aspect_ratio)

        print(f"Resizing person image and mask to: {target_width}x{target_height}")
        try:
            # Resize images while preserving aspect ratio
            person_image = resize_image(HWC3(person_image), target_width, target_height)
            inpaint_mask = resize_image(HWC3(inpaint_mask), target_width, target_height)
            print("Person image and mask resized successfully.")
        except Exception as e:
            print(f"Error occurred while resizing person image and mask: {str(e)}")
            raise e

        # Set the aspect ratio for the model
        aspect_ratio = f"{target_width}×{target_height}"

        print("Preparing arguments for image generation task...")
        try:
            # Prepare arguments for the image generation task
            args = [
                True,  # Input image checkbox
                prompt,  # Prompt for generating garment
                modules1.config.default_prompt_negative,  # Negative prompt
                False,  # Advanced checkbox
                modules1.config.default_styles,  # Style selections
                flags.Performance.QUALITY.value,  # Performance selection
                aspect_ratio,  # Aspect ratio selection
                1,  # Image number
                modules1.config.default_output_format,  # Output format
                np.random.randint(constants.MIN_SEED, constants.MAX_SEED),  # Random seed
                modules1.config.default_sample_sharpness,  # Sharpness
                modules1.config.default_cfg_scale,  # Guidance scale
                modules1.config.default_base_model_name,  # Base model
                modules1.config.default_refiner_model_name,  # Refiner model
                modules1.config.default_refiner_switch,  # Refiner switch
            ]

            # Add LoRA arguments
            for lora in modules1.config.default_loras:
                args.extend(lora)

            args.extend([
                False,  # Current tab (set to True for inpainting)
                "inpaint",  # Inpaint mode
                flags.disabled,  # UOV method
                None,  # UOV input image
                [],  # Outpaint selections
                {'image': person_image, 'mask': inpaint_mask},  # Inpaint input image
                prompt,  # Inpaint additional prompt
                inpaint_mask,  # Inpaint mask image
                True,  # Disable preview
                True,  # Disable intermediate results
                modules1.config.default_black_out_nsfw,  # Black out NSFW
                1.5,  # Positive ADM guidance
                0.8,  # Negative ADM guidance
                0.3,  # ADM guidance end
                modules1.config.default_cfg_tsnr,  # CFG mimicking from TSNR
                modules1.config.default_sampler,  # Sampler name
                modules1.config.default_scheduler,  # Scheduler name
                -1,  # Overwrite step
                -1,  # Overwrite switch
                orig_person_w,  # Overwrite width
                orig_person_h,  # Overwrite height
                -1,  # Overwrite vary strength
                modules1.config.default_overwrite_upscale,  # Overwrite upscale strength
                False,  # Mixing image prompt and vary upscale
                False,  # Mixing image prompt and inpaint
                True,  # Debugging CN preprocessor
                True,  # Skipping CN preprocessor
                100,  # Canny low threshold
                200,  # Canny high threshold
                flags.refiner_swap_method,  # Refiner swap method
                0.5,  # ControlNet softness
                False,  # FreeU enabled
                1.0,  # FreeU b1
                1.0,  # FreeU b2
                1.0,  # FreeU s1
                1.0,  # FreeU s2
                False,  # Debugging inpaint preprocessor
                False,  # Inpaint disable initial latent
                modules1.config.default_inpaint_engine_version,  # Inpaint engine
                1.0,  # Inpaint strength
                0.618,  # Inpaint respective field
                False,  # Inpaint mask upload checkbox
                False,  # Invert mask checkbox
                0,  # Inpaint erode or dilate
                modules1.config.default_save_metadata_to_images,  # Save metadata to images
                modules1.config.default_metadata_scheme,  # Metadata scheme
            ])
            print("Arguments prepared successfully.")
        except Exception as e:
            print(f"Error occurred while preparing arguments for image generation task: {str(e)}")
            raise e

        print("Creating image generation task...")
        try:
            # Create and append the image generation task
            task = worker.AsyncTask(args=args)
            worker.async_tasks.append(task)
            print("Image generation task created and appended successfully.")
        except Exception as e:
            print(f"Error occurred while creating and appending image generation task: {str(e)}")
            raise e

        print("Waiting for task to start processing...")
        try:
            # Wait for the task to start and finish processing
            while not task.processing:
                time.sleep(0.1)
            print("Task started processing.")
        except Exception as e:
            print(f"Error occurred while waiting for task to start processing: {str(e)}")
            raise e

        print("Waiting for task to finish processing...")
        try:
            while task.processing:
                time.sleep(0.1)
            print("Task finished processing.")
        except Exception as e:
            print(f"Error occurred while waiting for task to finish processing: {str(e)}")
            raise e

        print("Checking if results were generated successfully...")
        try:
            # Check if results were generated successfully
            if task.results and isinstance(task.results, list) and len(task.results) > 0:
                result_path = task.results[0]

                if output_path:
                    print(f"Saving result image to: {output_path}")
                    try:
                        os.rename(result_path, output_path)
                        result_path = output_path
                    except Exception as e:
                        print(f"Error occurred while saving result image: {str(e)}")
                        raise e
                print(f"Virtual try-on completed successfully. Result saved at: {result_path}")
                return result_path
            else:
                print("No results generated.")
                return None
        except Exception as e:
            print(f"Error occurred while checking if results were generated successfully: {str(e)}")
            raise e

    except Exception as e:
        print(f"Error occurred in virtual_try_on:")
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    input_image_path = "tryon/in_im/dc85c19de00a340b6bdd94c99bfd623b.jpg"
    output_image_path = "tryon/in_im/image.jpg"
    prompt = "Remove clothes, full naked, straight pose standing posing forward straight, perfect anatomy"
    category = "dresses"  # or "lower_body" or "upper_body"

    result = virtual_try_on(input_image_path, prompt, category, output_image_path)
    if result:
        print(f"Virtual try-on completed successfully. Result saved at: {result}")
    else:
        print("Virtual try-on failed.")
