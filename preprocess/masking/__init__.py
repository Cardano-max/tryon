from .mask_generator import MaskGenerator
from .factory import MaskFactory

class Masking:
    def __init__(self):
        self.factory = MaskFactory()
        self.mask_generator = MaskGenerator()

    def get_mask(self, image, category='dresses', model_type='hd'):
        adapter = self.factory.get_mask_adapter(model_type)
        return adapter.generate_mask(image, category)