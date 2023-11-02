# ptufile/__init__.py
# pylint: skip-file

from .ptufile import *
from .ptufile import __all__, __doc__, __version__

# constants are repeated for documentation

__version__ = __version__
"""Ptufile version string."""

T2_RECORD_DTYPE = T2_RECORD_DTYPE
"""Numpy dtype of decoded T2 records."""

T3_RECORD_DTYPE = T3_RECORD_DTYPE
"""Numpy dtype of decoded T3 records."""
