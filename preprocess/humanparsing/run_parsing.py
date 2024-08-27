import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference

class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        self.session = None
        self.lip_session = None
        
        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.add_session_config_entry('gpu_id', str(gpu_id))
            
            parsing_atr_path = os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_atr.onnx')
            parsing_lip_path = os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_lip.onnx')
            
            if os.path.exists(parsing_atr_path) and os.path.exists(parsing_lip_path):
                self.session = ort.InferenceSession(parsing_atr_path, sess_options=session_options, providers=['CPUExecutionProvider'])
                self.lip_session = ort.InferenceSession(parsing_lip_path, sess_options=session_options, providers=['CPUExecutionProvider'])
            else:
                print("ONNX model files not found. Falling back to dummy parsing.")
        except Exception as e:
            print(f"Error initializing ONNX sessions: {e}")
            print("Falling back to dummy parsing.")

    def dummy_parse(self, input_image):
        # Create a dummy parsed image and face mask
        dummy_parsed = np.zeros_like(input_image)
        dummy_face_mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.float32)
        return Image.fromarray(dummy_parsed), torch.from_numpy(dummy_face_mask)

    def __call__(self, input_image):
        if self.session is None or self.lip_session is None:
            return self.dummy_parse(input_image)
        
        try:
            parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
            return parsed_image, face_mask
        except Exception as e:
            print(f"Error during parsing: {e}")
            return self.dummy_parse(input_image)