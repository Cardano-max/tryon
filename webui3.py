import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
from preprocess.masking import Masking
import uuid
import signal
from masking_module.masking_module import generate_mask as hf_generate_mask


print(sys.path)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Initialize Masker
masker = Masking()

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Processing timed out")

signal.signal(signal.SIGALRM, timeout_handler)

def generate_mask(person_image, category="dresses"):
    if not isinstance(person_image, Image.Image):
        person_image = Image.fromarray(person_image)
    
    print("Generating mask...")
    try:
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_prompt = f"A person wearing {category} clothing"
        inpaint_mask = hf_generate_mask(
            image_input=person_image,
            image_url=None,
            task_prompt=task_prompt,
            text_prompt=text_prompt,
            dilate=10,
            merge_masks=True,
            return_rectangles=False,
            invert_mask=False
        )
        print("Mask generated successfully.")
        if inpaint_mask and len(inpaint_mask) > 0:
            return inpaint_mask[0]  # Return the first mask
        else:
            raise ValueError("No mask generated")
    except Exception as e:
        print(f"Error occurred while generating mask: {str(e)}")
        raise e

def virtual_try_on(person_image_path, prompt, category="dresses", output_path=None):
    try:
        print(f"Loading person image from: {person_image_path}")
        person_image = Image.open(person_image_path).convert('RGB')
        person_image = np.array(person_image)
        print("Person image loaded successfully.")

        inpaint_mask = generate_mask(person_image, category)

        orig_person_h, orig_person_w = person_image.shape[:2]
        person_aspect_ratio = orig_person_h / orig_person_w
        target_width = 1024
        target_height = int(target_width * person_aspect_ratio)
        target_height = min(target_height, 1024)
        target_width = int(target_height / person_aspect_ratio)

        print(f"Resizing person image and mask to: {target_width}x{target_height}")
        person_image = resize_image(HWC3(person_image), target_width, target_height)
        inpaint_mask = resize_image(HWC3(inpaint_mask), target_width, target_height)
        print("Person image and mask resized successfully.")

        aspect_ratio = f"{target_width}×{target_height}"

        print("Preparing arguments for image generation task...")
        args = [
            True,  # Input image checkbox
            prompt,  # Prompt for generating garment
            modules1.config.default_prompt_negative,  # Negative prompt
            True,  # Advanced checkbox
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
        
        for lora in modules1.config.default_loras:
            args.extend(lora)

        args.extend([
            True,  # Current tab (set to True for inpainting)
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
            target_width,  # Overwrite width
            target_height,  # Overwrite height
            -1,  # Overwrite vary strength
            modules1.config.default_overwrite_upscale,  # Overwrite upscale strength
            False,  # Mixing image prompt and vary upscale
            True,  # Mixing image prompt and inpaint
            False,  # Debugging CN preprocessor
            False,  # Skipping CN preprocessor
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
            True,  # Inpaint disable initial latent
            modules1.config.default_inpaint_engine_version,  # Inpaint engine
            1.0,  # Inpaint strength
            0.618,  # Inpaint respective field
            True,  # Inpaint mask upload checkbox
            False,  # Invert mask checkbox
            0,  # Inpaint erode or dilate
            modules1.config.default_save_metadata_to_images,  # Save metadata to images
            modules1.config.default_metadata_scheme,  # Metadata scheme
        ])
        print("Arguments prepared successfully.")

        print("Creating image generation task...")
        task = worker.AsyncTask(args=args)
        worker.async_tasks.append(task)
        print("Image generation task created and appended successfully.")

        print("Waiting for task to start processing...")
        timeout = 300  # 5 minutes timeout
        signal.alarm(timeout)
        try:
            start_time = time.time()
            while not task.processing:
                time.sleep(0.1)
                if time.time() - start_time > timeout:
                    raise TimeoutException("Task failed to start processing within the timeout period.")
            print("Task started processing.")
            
            while task.processing:
                time.sleep(0.1)
                if time.time() - start_time > timeout:
                    raise TimeoutException("Task processing exceeded the timeout period.")
            print("Task finished processing.")
        finally:
            signal.alarm(0)

        print("Checking if results were generated successfully...")
        if task.results and isinstance(task.results, list) and len(task.results) > 0:
            result_path = task.results[0]
            if output_path:
                print(f"Saving result image to: {output_path}")
                os.rename(result_path, output_path)
                result_path = output_path
            print(f"Virtual try-on completed successfully. Result saved at: {result_path}")
            return result_path
        else:
            print("No results generated.")
            return None

    except TimeoutException as e:
        print(f"Timeout occurred: {str(e)}")
        return None
    except Exception as e:
        print(f"Error occurred in virtual_try_on:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    input_directory = "/Tes"
    output_directory = "results"
    prompt = "Remove clothes, full naked, follow pose correctly, perfect anatomy"
    category = "dresses"

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all JPEG files in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".jpeg") or filename.lower().endswith(".jpg"):
            input_image_path = os.path.join(input_directory, filename)
            output_image_path = os.path.join(output_directory, f"result_{filename}")

            print(f"Processing image: {input_image_path}")
            result = virtual_try_on(input_image_path, prompt, category, output_image_path)
            if result:
                print(f"Virtual try-on completed successfully. Result saved at: {result}")
            else:
                print("Virtual try-on failed.")