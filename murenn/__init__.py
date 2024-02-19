"""
MuReNN: MultiResolution Neural Networks
=======================================

murenn is a Python module which integrates efficient operators
for multiresolution analysis into a differentiable computing
framework.
"""

# List of top-level public names.
__all__ = ["DTCWT", "DTCWTForward"]


# Submodule imports
from .dtcwt.transform1d import DTCWTForward, DTCWTInverse
from .version import version as __version__

# PytW-like aliases
DTCWT = DTCWTForward
