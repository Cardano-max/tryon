import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from functools import wraps
from time import time
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from pathlib import Path
from skimage import measure, morphology
from scipy import ndimage

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

class Masking:
    def __init__(self):
        self.parsing_model = Parsing(-1)
        self.openpose_model = OpenPose(-1)
        self.label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17, "neck": 18
        }

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    @timing
    def get_mask(self, img, category='upper_body'):
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

        hand_mask = self.create_precise_hand_mask(img_np)
        arm_mask = np.isin(parse_array, [self.label_map["left_arm"], self.label_map["right_arm"]])
        
        # Combine hand and arm masks without dilation
        hand_arm_mask = np.logical_or(hand_mask, arm_mask)
        
        # Remove hand and arm regions from the garment mask
        mask = np.logical_and(mask, np.logical_not(hand_arm_mask))

        # Refine the mask
        mask = self.refine_mask(mask)
        mask = self.smooth_edges(mask, sigma=0.5)

        # Ensure the mask covers the full garment
        mask = self.fill_garment_gaps(mask, parse_array, category)

        # Reapply hand and arm mask to ensure they're not covered
        mask = np.logical_and(mask, np.logical_not(hand_arm_mask))

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(img.size, Image.LANCZOS)
        
        return np.array(mask_pil)

    def create_precise_hand_mask(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self.hand_landmarker.detect(mp_image)
        
        hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_points = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    hand_points.append([x, y])
                hand_points = np.array(hand_points, dtype=np.int32)
                cv2.fillPoly(hand_mask, [hand_points], 1)
        
        return hand_mask > 0

    def refine_mask(self, mask):
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Apply minimal morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours and keep only the largest one
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_refined = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_refined, [largest_contour], 0, 255, 1)  # Thin outline
            mask_refined = cv2.fillPoly(mask_refined, [largest_contour], 255)
        else:
            mask_refined = mask_uint8
        
        return mask_refined > 0

    def smooth_edges(self, mask, sigma=0.5):
        mask_float = mask.astype(float)
        mask_blurred = ndimage.gaussian_filter(mask_float, sigma=sigma)
        mask_smooth = (mask_blurred > 0.5).astype(np.uint8)
        return mask_smooth

    def fill_garment_gaps(self, mask, parse_array, category):
        if category == 'upper_body':
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"]]
        elif category == 'lower_body':
            garment_labels = [self.label_map["pants"], self.label_map["skirt"]]
        else:  # dresses
            garment_labels = [self.label_map["upper_clothes"], self.label_map["dress"], 
                              self.label_map["pants"], self.label_map["skirt"]]
        
        garment_region = np.isin(parse_array, garment_labels)
        
        # Use the garment region to fill gaps in the mask
        filled_mask = np.logical_or(mask, garment_region)
        
        # Remove small isolated regions
        filled_mask = self.remove_small_regions(filled_mask)
        
        return filled_mask

    def remove_small_regions(self, mask, min_size=100):
        labeled, num_features = measure.label(mask, return_num=True)
        for i in range(1, num_features + 1):
            region = (labeled == i)
            if np.sum(region) < min_size:
                mask[region] = 0
        return mask