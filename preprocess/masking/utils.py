import numpy as np
import cv2

def extend_arm_mask(wrist, elbow, scale):
    return elbow + scale * (wrist - elbow)

def hole_fill(img):
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)
        for j in range(len(area)):
            if j != i and area[i] > 2000:
                cv2.drawContours(refine_mask, contours, j, color=255, thickness=-1)
    return refine_mask

def refine_hole(parsing_result_filled, parsing_result, arm_mask):
    filled_hole = cv2.bitwise_and(np.where(parsing_result_filled == 4, 255, 0),
                                  np.where(parsing_result != 4, 255, 0)) - arm_mask * 255
    contours, hierarchy = cv2.findContours(filled_hole, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    refine_hole_mask = np.zeros_like(parsing_result).astype(np.uint8)
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], True)
        if abs(a) > 2000:
            cv2.drawContours(refine_hole_mask, contours, i, color=255, thickness=-1)
    return refine_hole_mask + arm_mask