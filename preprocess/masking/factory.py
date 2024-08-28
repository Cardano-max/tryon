from .adapters.hd_adapter import HDMaskAdapter
from .adapters.dc_adapter import DCMaskAdapter

class MaskFactory:
    @staticmethod
    def get_mask_adapter(model_type):
        if model_type == 'hd':
            return HDMaskAdapter()
        elif model_type == 'dc':
            return DCMaskAdapter()
        else:
            raise ValueError("Invalid model type")