from __future__ import annotations

import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class EssentialMatrix:
    """
    EssentialMatrix is a class that represents an essential matrix.

    Attributes:
    ----------
    value: np.ndarray
        The essential matrix with shape (3, 3).
    """
    value: np.ndarray

    def __post_init__(self):
        if self.value.shape != (3, 3):
            raise ValueError("Essential matrix must be a 3x3 matrix")

