from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .method import PerspectiveTransformationMethod
from .mixin.factory import PerspectiveMatrixFactoryMixin
from .mixin.transform import PerspectiveMatrixTransformMixin


@dataclass
class PerspectiveMatrix(
    PerspectiveMatrixFactoryMixin,
    PerspectiveMatrixTransformMixin,
    ABC,
):
    """
    Container class for perspective transformation matrices.

    This class provides a unified interface for both affine and homography transformations,
    automatically selecting the appropriate matrix type based on the input.

    Attributes:
    ----------
    value: np.ndarray
        The perspective transformation matrix with shape (N, M).
    """
    value: np.ndarray

    @property
    @abstractmethod
    def translation(self) -> np.ndarray:
        """
        Get the translation component of the transformation.

        Returns
        -------
        np.ndarray:
            The translation component of the transformation with shape (2,).
        """

    @property
    def rotation(self) -> float:
        """
        Return the rotation angle in radians, normalized by scale.

        Calculated from the linear transformation components:
        a, b = H[0,0], H[0,1] (or A[0,0], A[0,1])
        c, d = H[1,0], H[1,1] (or A[1,0], A[1,1])
        """
        a, _ = self.value[0, :2]  # a, b: float
        c, _ = self.value[1, :2]  # c, d: float
        sx = np.hypot(a, c)
        if sx == 0:
            return 0
        return np.arctan2(c / sx, a / sx)

    @property
    def scale(self) -> np.ndarray:
        """
        Return the scale factors (sx, sy).

        Calculated from the linear transformation components:
        sx = sqrt(a² + c²), sy = sqrt(b² + d²)

        Returns
        -------
        np.ndarray:
            The scale factors with shape (2,).
        """
        a, b = self.value[0, :2]
        c, d = self.value[1, :2]
        sx = np.hypot(a, c)
        sy = np.hypot(b, d)
        return np.array([sx, sy])

    @property
    @abstractmethod
    def shear(self) -> np.ndarray:
        """
        Get the shear factors.

        Returns
        -------
        np.ndarray:
            The shear factors with shape (2,).
        """

    @property
    @abstractmethod
    def has_perspective(self) -> bool:
        """
        Check if the transformation has perspective components.

        Returns
        -------
        bool:
            True if the transformation has perspective components, False otherwise.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value.shape={self.value.shape}, transform_type={self.transform_type})"

    @property
    def column_vector(self) -> np.ndarray:
        """
        Return the column vector of the matrix.

        Returns
        -------
        np.ndarray:
            The column vector of the matrix with shape (N, 1).
        """
        return self.value.reshape(-1, 1)

    @property
    def row_vector(self) -> np.ndarray:
        """
        Return the row vector of the matrix.

        Returns
        -------
        np.ndarray:
            The row vector of the matrix with shape (1, N).
        """
        return self.value.reshape(1, -1)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the matrix.

        Returns
        -------
        tuple[int, int]:
            The shape of the matrix.
        """
        return self.value.shape

    @property
    def flatten(self) -> np.ndarray:
        """
        Return the flattened matrix.

        Returns
        -------
        np.ndarray:
            The flattened matrix with shape (6,) or (9,).
        """
        return self.value.flatten()

    @property
    @abstractmethod
    def transform_type(self) -> PerspectiveTransformationMethod:
        """
        Return the type of the transformation.

        Returns
        -------
        PerspectiveTransformationMethod:
            The type of the transformation.
        """
