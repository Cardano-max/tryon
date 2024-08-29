import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from ..humanparsing.run_parsing import Parsing
from ..openpose.run_openpose import OpenPose
from .utils import timing, refine_mask, create_hand_mask, hole_fill, extend_arm_mask

class MaskGenerator:
    def __init__(self):
        self.parsing_model = Parsing(-1)
        self.openpose_model = OpenPose(-1)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17, "neck": 18
        }

    @timing
    def generate_mask(self, img, category='upper_body', model_type='hd'):
        # Resize image to 384x512 for processing
        img_resized = img.resize((384, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        # Get human parsing result
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        # Get pose estimation
        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        # Create initial mask based on category
        if category == 'upper_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'dresses':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                         self.label_map["pants"], self.label_map["skirt"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        # Create arm mask
        arm_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"]])

        # Create hand mask using MediaPipe
        hand_mask = create_hand_mask(img_np, self.hands)

        # Combine arm and hand mask
        arm_hand_mask = np.logical_or(arm_mask, hand_mask)

        # Remove arms and hands from the mask
        mask = np.logical_and(mask, np.logical_not(arm_hand_mask))

        # Refine the mask
        mask = refine_mask(mask)

        # Resize mask back to original image size
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return np.array(mask_pil)