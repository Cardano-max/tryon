import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
import torch
import numpy as np
from PIL import Image
import logging

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.add_session_config_entry('gpu_id', str(gpu_id))

        self.session = None
        self.lip_session = None

        try:
            self.session = ort.InferenceSession(
                os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_atr.onnx'),
                sess_options=session_options, providers=['CPUExecutionProvider']
            )
            self.lip_session = ort.InferenceSession(
                os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_lip.onnx'),
                sess_options=session_options, providers=['CPUExecutionProvider']
            )
            logger.info("ONNX models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ONNX models: {str(e)}")
            logger.info("Falling back to dummy parsing method")

    def dummy_parse(self, input_image):
        # Create a simple segmentation mask
        if isinstance(input_image, str):
            input_image = Image.open(input_image)
        elif isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        width, height = input_image.size
        mask = Image.new('L', (width, height), 0)
        
        # Simple segmentation: top half is "upper body", bottom half is "lower body"
        for y in range(height):
            for x in range(width):
                if y < height // 2:
                    mask.putpixel((x, y), 4)  # Upper body
                else:
                    mask.putpixel((x, y), 6)  # Lower body

        face_mask = torch.zeros((height, width), dtype=torch.float32)
        # Simple face detection: assume face is in the top-center of the image
        face_top = height // 8
        face_bottom = height // 3
        face_left = width // 3
        face_right = 2 * width // 3
        face_mask[face_top:face_bottom, face_left:face_right] = 1.0

        return mask, face_mask

    def __call__(self, input_image):
        if self.session is None or self.lip_session is None:
            logger.info("Using dummy parsing method")
            return self.dummy_parse(input_image)
        
        try:
            parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
            return parsed_image, face_mask
        except Exception as e:
            logger.error(f"Error during ONNX inference: {str(e)}")
            logger.info("Falling back to dummy parsing method")
            return self.dummy_parse(input_image)