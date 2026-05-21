from __future__ import annotations

from functools import cached_property
from typing import Protocol

import numpy as np


class HomographyInverseProtocol(Protocol):
    """Homography container that exposes a cached matrix inverse."""

    value: np.ndarray

    @cached_property
    def inverse(self) -> np.ndarray: ...
