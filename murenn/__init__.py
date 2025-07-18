"""
MuReNN: MultiResolution Neural Networks
=======================================

murenn is a Python module which integrates efficient operators
for multiresolution analysis into a differentiable computing
framework.
"""

# List of top-level public names.
__all__ = ["DTCWT", "DTCWTDirect", "DTCWTInverse"]


# Submodule imports
from .dtcwt.transform1d import DTCWTDirect, DTCWTInverse
from .dtcwt.udtcwt1d import UDTCWTDirect
from .version import version as __version__
from .dtcwt.nn_antialiasing import MuReNNDirect

# PytW-like aliases
DTCWT = DTCWTDirect
IDTCWT = DTCWTInverse
UDTCWT = UDTCWTDirect