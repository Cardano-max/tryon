import numpy as np
from PIL import Image
from pathlib import Path
from functools import wraps
from time import time
import torch

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

    @timing
    def get_mask(self, img, category='full_body'):
        # Resize image to 512x512
        img_resized = img.resize((512, 512), Image.LANCZOS)
        
        # Use SegBody to create the mask
        _, mask_image = segment_body(img_resized, face=False)
        
        # Convert mask to binary (0 or 255)
        mask_array = np.array(mask_image)
        binary_mask = (mask_array > 128).astype(np.uint8) * 255
        
        # Create binary mask PIL Image
        mask_binary = Image.fromarray(binary_mask)
        
        # Create grayscale mask
        grayscale_mask = (mask_array > 128).astype(np.uint8) * 127
        mask_gray = Image.fromarray(grayscale_mask)
        
        # Resize masks back to original image size
        mask_binary = mask_binary.resize(img.size, Image.LANCZOS)
        mask_gray = mask_gray.resize(img.size, Image.LANCZOS)
        
        return mask_binary, mask_gray