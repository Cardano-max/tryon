import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from functools import wraps
from time import time
import torch
from scipy.ndimage import gaussian_filter

# Ensure torch uses CPU if CUDA is not available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import SegBody (assuming it's in the same directory)
from SegBody import segment_body

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} args:[{args}, {kw}] took: {te-ts:.4f} sec')
        return result
    return wrap

class Masking:
    def __init__(self):
        pass

    def dilate_mask(self, mask, dilation_percent):
        kernel_size = int(max(mask.shape) * dilation_percent / 100)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return dilated

    def smooth_edges(self, mask, sigma=1.0):
        return gaussian_filter(mask.astype(float), sigma=sigma)

    def create_gradient_border(self, mask, border_size):
        gradient = np.zeros_like(mask, dtype=np.float32)
        for i in range(border_size):
            weight = 1 - (i / border_size)
            gradient = np.maximum(gradient, cv2.dilate(mask, None) * weight)
        return gradient

    @timing
    def get_mask(self, img, category='full_body'):
        # Resize image to 512x512
        img_resized = img.resize((512, 512), Image.LANCZOS)
        
        # Use SegBody to create the mask
        _, mask_image = segment_body(img_resized, face=False)
        
        # Convert mask to binary (0 or 255)
        mask_array = np.array(mask_image)
        binary_mask = (mask_array > 128).astype(np.uint8) * 255
        
        # Dilate the mask (3-5% extra on edges towards outside)
        dilated_mask = self.dilate_mask(binary_mask, 5)
        
        # Smooth the edges
        smoothed_mask = self.smooth_edges(dilated_mask, sigma=2.0)
        
        # Create a gradient border for fading effect
        border_size = int(max(smoothed_mask.shape) * 0.05)  # 5% of image size
        gradient_border = self.create_gradient_border(smoothed_mask, border_size)
        
        # Combine smoothed mask with gradient border
        final_mask = np.clip(smoothed_mask + gradient_border, 0, 255).astype(np.uint8)
        
        # Create binary and grayscale masks
        mask_binary = Image.fromarray(final_mask)
        mask_gray = Image.fromarray((final_mask / 2).astype(np.uint8))
        
        # Resize masks back to original image size
        mask_binary = mask_binary.resize(img.size, Image.LANCZOS)
        mask_gray = mask_gray.resize(img.size, Image.LANCZOS)
        
        return mask_binary, mask_gray