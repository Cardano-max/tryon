import os
import torch
import numpy as np
import cv2
from .detectron2.config import get_cfg
from .detectron2.engine.defaults import DefaultPredictor
from .detectron2.structures import Instances
from .detectron2.utils.logger import setup_logger
from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.vis.extractor import DensePoseResultExtractor, DensePoseOutputsExtractor, create_extractor
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_results_textures import get_texture_atlas

def setup_config(config_fpath, model_fpath, args, opts):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.merge_from_list(args.opts)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.freeze()
    return cfg

class InferenceEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, image):
        with torch.no_grad():
            outputs = self.predictor(image)["instances"]
        return outputs

def process_outputs(outputs, extractor):
    if not isinstance(outputs, Instances):
        raise ValueError(f"Outputs must be an instance of Instances, got {type(outputs)}")
    
    if outputs.has("pred_densepose"):
        if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
            densepose_extractor = DensePoseResultExtractor()
        elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
            densepose_extractor = DensePoseOutputsExtractor()
        else:
            raise ValueError(f"Unknown DensePose output type: {type(outputs.pred_densepose)}")
        densepose_results = densepose_extractor(outputs)[0]
    else:
        densepose_results = None

    return extractor(outputs), densepose_results

def apply_densepose(image, config_path, model_path, visualizations="dp_segm"):
    logger = setup_logger(name="apply_net")
    
    cfg = setup_config(config_path, model_path, argparse.Namespace(opts=[]), [])
    inference_engine = InferenceEngine(cfg)

    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        raise ValueError("Could not read image")
    
    outputs = inference_engine(image)
    
    vis_specs = visualizations.split(",")
    visualizers = []
    extractors = []
    for vis_spec in vis_specs:
        texture_atlas = get_texture_atlas("", is_custom=False)
        vis = ShowAction.VISUALIZERS[vis_spec](
            cfg=cfg,
            texture_atlas=texture_atlas,
            texture_atlases_dict=None,
        )
        visualizers.append(vis)
        extractor = create_extractor(vis)
        extractors.append(extractor)
    visualizer = CompoundVisualizer(visualizers)
    extractor = CompoundExtractor(extractors)
    
    results, _ = process_outputs(outputs, extractor)
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_vis = visualizer.visualize(image_bgr, results)
    return image_vis