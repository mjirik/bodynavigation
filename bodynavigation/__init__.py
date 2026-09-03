__all__ = ["body_navigation"]
import numpy as np

# This project still uses these NumPy scalar aliases internally.  They were
# removed in NumPy 2, which is used by the PilsenPigs environment.
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

from .body_navigation import BodyNavigation

__version__ = "0.6.1"
