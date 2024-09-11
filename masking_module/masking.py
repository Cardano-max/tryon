import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import requests
from typing import Optional, List
import time

from .utils.florence import load_florence_model, run_florence_inference
from .utils.sam import load_sam_image_model, run_sam_inference
import supervision as sv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the paths for SAM model
SAM_CHECKPOINT_PATH = "masking_module/sam2/checkpoints/sam2_hiera_large.pt"
SAM_CONFIG_PATH = "masking_module/sam2/configs/sam2_hiera_l.yaml"

# Load models
FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE, config=SAM_CONFIG_PATH, checkpoint=SAM_CHECKPOINT_PATH)

class calculateDuration:
    def __init__(self, activity_name=""):
        self.activity_name = activity_name

    def __enter__(self):
        self.start_time = time.time()
        self.start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))
        print(f"Activity: {self.activity_name}, Start time: {self.start_time_formatted}")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time))
        
        if self.activity_name:
            print(f"Elapsed time for {self.activity_name}: {self.elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {self.elapsed_time:.6f} seconds")
        
        print(f"Activity: {self.activity_name}, End time: {self.start_time_formatted}")

def fetch_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error fetching image: {str(e)}")
        return None

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def generate_mask(
    image_input: Optional[Image.Image],
    image_url: Optional[str],
    task_prompt: str,
    text_prompt: Optional[str] = None,
    dilate: int = 0,
    merge_masks: bool = False,
    return_rectangles: bool = False,
    invert_mask: bool = False
) -> Optional[List[np.ndarray]]:
    
    if not image_input and not image_url:
        print("Please provide either an image or an image URL.")
        return None
    
    if not task_prompt:
        print("Please enter a task prompt.")
        return None
   
    if image_url:
        with calculateDuration("Download Image"):
            print("start to fetch image from url", image_url)
            image_input = fetch_image_from_url(image_url)
            if image_input is None:
                return None
            print("fetch image success")
    
    # start to parse prompt
    with calculateDuration("FLORENCE"):
        print(task_prompt, text_prompt)
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=task_prompt,
            text=text_prompt
        )
    
    with calculateDuration("sv.Detections"):
        # start to detect
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
    
    images = []
    if return_rectangles:
        with calculateDuration("generate rectangle mask"):
            # create mask in rectangle
            (image_width, image_height) = image_input.size
            bboxes = detections.xyxy
            merge_mask_image = np.zeros((image_height, image_width), dtype=np.uint8)
            # sort from left to right
            bboxes = sorted(bboxes, key=lambda bbox: bbox[0])
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(merge_mask_image, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
                clip_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                cv2.rectangle(clip_mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
                images.append(clip_mask)
            if merge_masks:
                images = [merge_mask_image] + images
    else:
        with calculateDuration("generate segment mask"):
            # using sam generate segments images        
            detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
            if len(detections) == 0:
                print("No objects detected.")
                return None
            print("mask generated:", len(detections.mask))
            kernel_size = dilate
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            for i in range(len(detections.mask)):
                mask = detections.mask[i].astype(np.uint8) * 255
                if dilate > 0:
                    mask = cv2.dilate(mask, kernel, iterations=1)
                images.append(mask)

            if merge_masks:
                merged_mask = np.zeros_like(images[0], dtype=np.uint8)
                for mask in images:
                    merged_mask = cv2.bitwise_or(merged_mask, mask)
                images = [merged_mask]
    
    if invert_mask:
        with calculateDuration("invert mask colors"):
            images = [cv2.bitwise_not(mask) for mask in images]

    return images

# Usage example
if __name__ == "__main__":
    # Test with an image file
    image_path = "path/to/your/image.jpg"
    image = Image.open(image_path)
    
    # Test with an image URL
    # image_url = "https://example.com/image.jpg"
    # image = None
    
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_prompt = "A person wearing a red shirt"
    
    masks = generate_mask(
        image_input=image,
        image_url=None,  # Set to None if using image_input
        task_prompt=task_prompt,
        text_prompt=text_prompt,
        dilate=10,
        merge_masks=False,
        return_rectangles=False,
        invert_mask=False
    )
    
    if masks:
        for i, mask in enumerate(masks):
            Image.fromarray(mask).save(f"output_mask_{i}.png")
        print(f"Generated {len(masks)} masks")
    else:
        print("Mask generation failed")