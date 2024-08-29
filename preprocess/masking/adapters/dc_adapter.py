from .base_adapter import BaseMaskAdapter
from ..mask_generator import MaskGenerator

class DCMaskAdapter(BaseMaskAdapter):
    def __init__(self):
        self.mask_generator = MaskGenerator()

    def generate_mask(self, image, category):
        return self.mask_generator.generate_mask(image, category, 'dc')