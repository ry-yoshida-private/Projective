from __future__ import annotations

from typing import Protocol

import numpy as np


class HomographyInverseProtocol(Protocol):
    """Homography container that exposes a cached matrix inverse."""

    value: np.ndarray

    @property
    def inverse(self) -> np.ndarray: ...
