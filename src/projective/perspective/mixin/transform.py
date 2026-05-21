from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..matrix import PerspectiveMatrix


class PerspectiveTransformMixin(ABC):
    """Transform the matrix container itself (not external point coordinates)."""

    value: np.ndarray

    @property
    def column_vector(self) -> np.ndarray:
        """
        Return the column vector of the matrix.

        Returns
        -------
        np.ndarray
            Column vector with shape (value.size, 1).
        """
        return self.value.reshape(-1, 1)

    @property
    def row_vector(self) -> np.ndarray:
        """
        Return the row vector of the matrix.

        Returns
        -------
        np.ndarray
            Row vector with shape (1, value.size).
        """
        return self.value.reshape(1, -1)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the matrix.

        Returns
        -------
        tuple[int, int]
            Two-dimensional shape of value.
        """
        return self.value.shape

    @property
    def flatten(self) -> np.ndarray:
        """
        Return the flattened matrix.

        Returns
        -------
        np.ndarray
            One-dimensional view with shape (value.size,).
        """
        return self.value.flatten()

    @abstractmethod
    def scale_correction(
        self,
        scale: float,
    ) -> PerspectiveMatrix:
        """
        Correct the scale of the perspective matrix.

        Parameters
        ----------
        scale : float
            Scale factor.

        Returns
        -------
        PerspectiveMatrix
            The perspective matrix with corrected scale.
        """
