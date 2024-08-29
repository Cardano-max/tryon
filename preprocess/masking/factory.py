from .adapters import HDMaskAdapter, DCMaskAdapter

class MaskFactory:
    @staticmethod
    def get_mask_adapter(model_type):
        if model_type == 'hd':
            return HDMaskAdapter()
        elif model_type == 'dc':
            return DCMaskAdapter()
        else:
            raise ValueError("Invalid model type. Choose 'hd' or 'dc'.")