from typing import Any

import numpy as np
import supervision as sv
import torch
from PIL import Image
from ..sam2.build_sam import build_sam2, build_sam2_video_predictor
from ..sam2.sam2_image_predictor import SAM2ImagePredictor

# SAM_CHECKPOINT = "checkpoints/sam2_hiera_small.pt"
# SAM_CONFIG = "sam2_hiera_s.yaml"
SAM_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
SAM_CONFIG = "sam2_hiera_l.yaml"


def load_sam_image_model(
    device: torch.device,
    config: str,
    checkpoint: str
) -> SAM2ImagePredictor:
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(sam_model=model)


def load_sam_video_model(
    device: torch.device,
    config: str = SAM_CONFIG,
    checkpoint: str = SAM_CHECKPOINT
) -> Any:
    return build_sam2_video_predictor(config, checkpoint, device=device)


def run_sam_inference(
    model: Any,
    image: Image,
    detections: sv.Detections
) -> sv.Detections:
    image = np.array(image.convert("RGB"))
    model.set_image(image)
    # from left to right
    bboxes = detections.xyxy
    bboxes = sorted(bboxes, key=lambda bbox: bbox[0])
    mask, score, _ = model.predict(box=bboxes, multimask_output=False)

    if mask is None:
        print("SAM model returned None for mask. Using fallback method.")
        # Create a simple rectangular mask as a fallback
        mask = np.zeros(image.shape[:2], dtype=bool)
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            mask[y1:y2, x1:x2] = True
    else:
        # dirty fix; remove this later
        if len(mask.shape) == 4:
            mask = np.squeeze(mask)

    detections.mask = mask.astype(bool)
    return detections
