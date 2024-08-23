# Copyright (c) Facebook, Inc. and its affiliates.

from .utils.env import setup_environment
from . import config
from . import data
from . import engine
from . import structures
from . import utils

setup_environment()


# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
__version__ = "0.6"
