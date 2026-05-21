from __future__ import annotations

from typing import Protocol

import numpy as np


class HomographyMatrixValueProtocol(Protocol):
    """Protocol for homography matrix instances."""

    value: np.ndarray
