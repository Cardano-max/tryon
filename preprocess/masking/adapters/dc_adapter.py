from .base_adapter import BaseMaskAdapter
from ..mask_generator import MaskGenerator

class DCMaskAdapter(BaseMaskAdapter):
    def generate_mask(self, category, model_parse, keypoint, width, height):
        return MaskGenerator.generate_mask('dc', category, model_parse, keypoint, width, height)