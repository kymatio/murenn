"""
MuReNN: MultiResolution Neural Networks
=======================================

murenn is a Python module which integrates efficient operators
for multiresolution analysis into a differentiable computing
framework.
"""

# List of top-level public names.
__all__ = ["DTCWT", "DTCWTDirect", "DTCWTInverse"]
__all__ = ["DTCWT", "DTCWTDirect"]


# Submodule imports
from .dtcwt.transform1d import DTCWTDirect, DTCWTInverse
from .dtcwt.transform1d import DTCWTDirect, DTCWTInverse
from .version import version as __version__

# PytW-like aliases
DTCWT = DTCWTDirect
DTCWT = DTCWTDirect
