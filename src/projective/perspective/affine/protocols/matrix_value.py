from __future__ import annotations

from typing import Protocol

import numpy as np


class AffineMatrixValueProtocol(Protocol):
    """Protocol for affine matrix instances."""

    value: np.ndarray
