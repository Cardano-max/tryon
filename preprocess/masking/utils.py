import numpy as np
import cv2
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

def create_hand_mask(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    hand_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                cv2.circle(hand_mask, (x, y), 15, 255, -1)
    
    return hand_mask > 0

def refine_mask(mask):
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

def extend_arm_mask(wrist, elbow, scale):
    return elbow + scale * (wrist - elbow)

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst