from fvcore.common.config import CfgNode as _CfgNode

class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but with different
    behavior in `clone()` that tries to maintain backward compatibility:
    """

    def clone(self):
        return type(self)(self)

    # Add any other methods or attributes that are used in the code

def get_cfg():
    """
    Get a copy of the default config.
    """
    from .defaults import _C
    return _C.clone()

# Add any other necessary functions or classes