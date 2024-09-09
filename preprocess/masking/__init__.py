import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from ..humanparsing.run_parsing import Parsing
from ..openpose.run_openpose import OpenPose

class Masking:
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

    def get_mask(self, img, category='dresses'):
        img_resized = img.resize((384, 512), Image.LANCZOS)
        img_np = np.array(img_resized)
        
        parse_result, _ = self.parsing_model(img_resized)
        parse_array = np.array(parse_result)

        keypoints = self.openpose_model(img_resized)
        pose_data = np.array(keypoints["pose_keypoints_2d"]).reshape((-1, 2))

        if category == 'upper_body':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"]])
        elif category == 'lower_body':
            mask = np.isin(parse_array, [self.label_map["pants"], self.label_map["skirt"]])
        elif category == 'dresses':
            mask = np.isin(parse_array, [self.label_map["upper_clothes"], self.label_map["dress"], 
                                         self.label_map["pants"], self.label_map["skirt"]])
        else:
            raise ValueError("Invalid category. Choose 'upper_body', 'lower_body', or 'dresses'.")

        arm_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"]])
        hand_mask = self.create_hand_mask(img_np)
        arm_hand_mask = np.logical_or(arm_mask, hand_mask)
        mask = np.logical_and(mask, np.logical_not(arm_hand_mask))

        mask = self.refine_mask(mask)
        mask = self.smooth_edges(mask, sigma=1.0)
        mask = self.expand_mask(mask, expansion=3)

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return np.array(mask_pil)

    def create_hand_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    cv2.circle(hand_mask, (x, y), 15, 255, -1)
        
        return hand_mask > 0

    def refine_mask(self, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        kernel = np.ones((5,5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, -1)
        else:
            mask_refined = mask_uint8
        
        return mask_refined > 0

    def smooth_edges(self, mask, sigma=1.0):
        from scipy import ndimage
        mask_float = mask.astype(float)
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
        mask_smooth = (mask_blurred > 0.5).astype(np.uint8)
        return mask_smooth

    def expand_mask(self, mask, expansion=3):
        kernel = np.ones((expansion, expansion), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return expanded_mask > 0