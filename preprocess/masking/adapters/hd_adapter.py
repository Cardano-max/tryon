from .base_adapter import BaseMaskAdapter
from ..mask_generator import MaskGenerator

class HDMaskAdapter(BaseMaskAdapter):
    def __init__(self):
        self.mask_generator = MaskGenerator()

    def generate_mask(self, image, category):
        return self.mask_generator.generate_mask(image, category, 'hd')