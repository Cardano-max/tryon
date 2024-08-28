import numpy as np
from PIL import Image, ImageDraw
from .utils import extend_arm_mask, hole_fill, refine_mask, refine_hole

class MaskGenerator:
    @staticmethod
    def generate_mask(model_type, category, model_parse, keypoint, width=384, height=512):
        label_map = {
            "background": 0, "hat": 1, "hair": 2, "sunglasses": 3, "upper_clothes": 4,
            "skirt": 5, "pants": 6, "dress": 7, "belt": 8, "left_shoe": 9, "right_shoe": 10,
            "head": 11, "left_leg": 12, "right_leg": 13, "left_arm": 14, "right_arm": 15,
            "bag": 16, "scarf": 17
        }

        im_parse = model_parse.resize((width, height), Image.NEAREST)
        parse_array = np.array(im_parse)

        if model_type == 'hd':
            arm_width = 60
        elif model_type == 'dc':
            arm_width = 45
        else:
            raise ValueError("model_type must be 'hd' or 'dc'!")

        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 3).astype(np.float32) + \
                     (parse_array == 11).astype(np.float32)

        parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["hat"]).astype(np.float32) + \
                            (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                            (parse_array == label_map["bag"]).astype(np.float32)

        parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

        arms_left = (parse_array == 14).astype(np.float32)
        arms_right = (parse_array == 15).astype(np.float32)

        pose_data = keypoint["pose_keypoints_2d"]
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 2))

        im_arms_left = Image.new('L', (width, height))
        im_arms_right = Image.new('L', (width, height))
        arms_draw_left = ImageDraw.Draw(im_arms_left)
        arms_draw_right = ImageDraw.Draw(im_arms_right)

        if category == 'dresses' or category == 'upper_body':
            shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
            shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
            elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
            elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
            wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
            wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
            
            ARM_LINE_WIDTH = int(arm_width / 512 * height)
            
            if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                im_arms_right = arms_right
            else:
                wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
                arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

            if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                im_arms_left = arms_left
            else:
                wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
                arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

        parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
        
        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        
        inpaint_mask = 1 - parse_mask_total
        img = np.where(inpaint_mask, 255, 0)
        dst = hole_fill(img.astype(np.uint8))
        dst = refine_mask(dst)
        inpaint_mask = dst / 255 * 1

        mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
        mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

        return mask, mask_gray