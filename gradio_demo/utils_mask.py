import numpy as np
import cv2
from PIL import Image, ImageDraw
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

label_map = {
    "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
    "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
    "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
    "bag": 16, "scarf": 17, "neck": 18
}

class Masking:
    def __init__(self):
        pass

    @staticmethod
    def extend_arm_mask(wrist, elbow, scale):
        return elbow + scale * (wrist - elbow)

    @staticmethod
    def hole_fill(img):
        img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
        img_copy = img.copy()
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(img, mask, (0, 0), 255)
        img_inverse = cv2.bitwise_not(img)
        return cv2.bitwise_or(img_copy, img_inverse)

    @staticmethod
    def refine_mask(mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        area = [abs(cv2.contourArea(contour, True)) for contour in contours]
        refine_mask = np.zeros_like(mask).astype(np.uint8)
        if area:
            i = area.index(max(area))
            cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)
        return refine_mask

    @timing
    def get_mask(self, img, category='full_body', model_type='hd', keypoint=None, width=384, height=512):
        # Use SegBody to create the initial mask
        img_resized = img.resize((512, 512), Image.LANCZOS)
        _, mask_image = segment_body(img_resized, face=False)
        
        # Convert mask to binary and resize
        mask_array = np.array(mask_image)
        binary_mask = (mask_array > 128).astype(np.uint8) * 255
        parse_array = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Initialize masks
        parse_head = np.zeros_like(parse_array)
        parser_mask_fixed = np.zeros_like(parse_array)
        parser_mask_changeable = (parse_array == 0).astype(np.float32)
        parse_mask = np.zeros_like(parse_array)

        # Apply pose-based refinements if keypoint data is provided
        if keypoint is not None:
            pose_data = np.array(keypoint["pose_keypoints_2d"]).reshape((-1, 2))
            pose_data = pose_data * np.array([width / 384, height / 512])

            # Create arm masks
            im_arms = Image.new('L', (width, height))
            arms_draw = ImageDraw.Draw(im_arms)
            
            if category in ['dresses', 'upper_body']:
                arm_width = 45 if model_type == 'dc' else 60
                ARM_LINE_WIDTH = int(arm_width / 512 * height)

                for side in ['right', 'left']:
                    shoulder = tuple(pose_data[2 if side == 'right' else 5])
                    elbow = tuple(pose_data[3 if side == 'right' else 6])
                    wrist = tuple(pose_data[4 if side == 'right' else 7])

                    if wrist[0] > 1 and wrist[1] > 1:
                        wrist = self.extend_arm_mask(wrist, elbow, 1.2)
                        arms_draw.line([shoulder, elbow, wrist], 'white', ARM_LINE_WIDTH, 'curve')
                        
                        size = [shoulder[0] - ARM_LINE_WIDTH // 2, shoulder[1] - ARM_LINE_WIDTH // 2,
                                shoulder[0] + ARM_LINE_WIDTH // 2, shoulder[1] + ARM_LINE_WIDTH // 2]
                        arms_draw.arc(size, 0, 360, 'white', ARM_LINE_WIDTH // 2)

            arm_mask = np.array(im_arms) / 255
            parse_mask = np.logical_or(parse_mask, arm_mask)

            # Handle neck area
            if category in ['dresses', 'upper_body']:
                neck_mask = cv2.dilate((parse_array == label_map["neck"]).astype(np.float32), 
                                       np.ones((5, 5), np.uint16), iterations=1)
                neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
                parse_mask = np.logical_or(parse_mask, neck_mask)

        # Combine masks
        parse_mask = cv2.dilate(parse_mask.astype(np.float32), np.ones((5, 5), np.uint16), iterations=5)
        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        
        # Final refinements
        inpaint_mask = 1 - parse_mask_total
        img = np.where(inpaint_mask, 255, 0)
        dst = self.hole_fill(img.astype(np.uint8))
        dst = self.refine_mask(dst)
        inpaint_mask = dst / 255

        mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
        mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

        # Resize masks to original image size
        original_size = img.size
        mask = mask.resize(original_size, Image.LANCZOS)
        mask_gray = mask_gray.resize(original_size, Image.LANCZOS)

        return mask, mask_gray

def process_images(input_folder, output_folder, category, model_type="hd"):
    masker = Masking()
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    image_files = list(Path(input_folder).glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')]
    
    for i, image_file in enumerate(image_files, 1):
        output_mask = Path(output_folder) / f"output_sharp_{i}.png"
        output_masked = Path(output_folder) / f"output_masked_image_white_{i}.png"
        
        print(f"Processing image {i}/{len(image_files)}: {image_file.name}")
        
        input_img = Image.open(image_file)
        
        # Assuming keypoint data is obtained here
        # In a real scenario, you would need to generate this data for each image
        keypoint = {
            "pose_keypoints_2d": [
                100, 100, 0.8,  # Example data for right shoulder
                150, 100, 0.8,  # Example data for right elbow
                200, 100, 0.8,  # Example data for right wrist
                300, 100, 0.8,  # Example data for left shoulder
                350, 100, 0.8,  # Example data for left elbow
                400, 100, 0.8,  # Example data for left wrist
            ]
        }
        
        mask, mask_gray = masker.get_mask(input_img, category, model_type, keypoint)
        
        mask.save(str(output_mask))
        
        white_bg = Image.new('RGB', input_img.size, (255, 255, 255))
        
        if input_img.mode != 'RGBA':
            input_img = input_img.convert('RGBA')
        
        masked_output = Image.composite(input_img, white_bg, mask)
        
        masked_output.save(str(output_masked))
        
        print(f"Mask saved to {output_mask}")
        print(f"Masked output with white background saved to {output_masked}")
        print()

if __name__ == "__main__":
    input_folder = Path("/path/to/input/folder")
    output_folder = Path("/path/to/output/folder")
    category = "dresses"  # Change to "upper_body", "lower_body", or "dresses" as needed
    
    process_images(str(input_folder), str(output_folder), category)