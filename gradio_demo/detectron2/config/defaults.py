from .config import CfgNode as CN

_C = CN()

# Add your default configuration here
_C.MODEL = CN()
_C.MODEL.ROI_DENSEPOSE_HEAD = CN()
# Add other configuration options as needed

def get_cfg():
    return _C.clone()