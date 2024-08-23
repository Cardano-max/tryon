import os
import yaml
from fvcore.common.config import CfgNode as _CfgNode

class CfgNode(_CfgNode):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)

    def merge_from_file(self, cfg_filename):
        assert os.path.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
        with open(cfg_filename, "r") as f:
            cfg = yaml.safe_load(f)
        self.merge_from_other_cfg(CfgNode(cfg))

    def merge_from_list(self, cfg_list):
        super().merge_from_list(cfg_list)

def get_cfg():
    from .defaults import _C
    return _C.clone()