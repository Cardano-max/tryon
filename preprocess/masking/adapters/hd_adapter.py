from .base_adapter import BaseMaskAdapter
from ..mask_generator import MaskGenerator

class HDMaskAdapter(BaseMaskAdapter):
    def generate_mask(self, category, model_parse, keypoint, width, height):
        return MaskGenerator.generate_mask('hd', category, model_parse, keypoint, width, height)