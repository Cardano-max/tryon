from .masking.factory import MaskFactory

class Masking:
    def __init__(self, model_type='hd'):
        self.mask_factory = MaskFactory()
        self.mask_adapter = self.mask_factory.get_mask_adapter(model_type)

    def get_mask(self, img, category='upper_body'):
        return self.mask_adapter.generate_mask(img, category)