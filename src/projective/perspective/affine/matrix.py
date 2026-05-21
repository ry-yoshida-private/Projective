from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ..matrix import PerspectiveMatrix
from ..method import PerspectiveTransformationMethod
from .mixin import (
    AffineFactoryMixin,
    AffineProjectiveMixin,
    AffineTransformMixin,
)


@dataclass
class AffineMatrix(
    AffineTransformMixin,
    AffineProjectiveMixin,
    AffineFactoryMixin,
    PerspectiveMatrix,
):
    """
    AffineMatrix is a class that represents an affine transformation matrix.

    Attributes:
    ----------
    value: np.ndarray
        The affine transformation matrix with shape (2, 3).
    """

    def __post_init__(self) -> None:
        """
        Post-init validation.

        Raises
        ------
        ValueError: If the affine transformation matrix is not a 2x3 matrix.
        """
        if self.value.shape != (2, 3):
            raise ValueError(f"Affine matrix must be a 2x3 matrix, got shape {self.value.shape}")

    @property
    def translation(self) -> np.ndarray:
        """
        Return the translation vector (tx, ty).

        Returns
        -------
        np.ndarray:
            The translation vector (tx, ty) with shape (2,).
        """
        return self.value[:, 2]

    @property
    def shear(self) -> np.ndarray:
        """
        Return shear factors along x and y axes.

        Calculated from the linear part of the affine matrix:
        [a b tx]
        [c d ty]

        Returns
        -------
        np.ndarray:
            The shear factors (sx, sy) with shape (2,).
        """
        a, b = self.value[0, :2]
        c, d = self.value[1, :2]
        sx2 = a**2 + c**2
        sy2 = b**2 + d**2

        shear_x = (a * b + c * d) / sx2 if not sx2 == 0 else 0
        shear_y = (a * b + c * d) / sy2 if not sy2 == 0 else 0
        return np.array([shear_x, shear_y])

    @property
    def has_perspective(self) -> bool:
        """
        Return True if the affine matrix has a perspective component.

        Returns
        -------
        bool:
            True if the affine matrix has a perspective component, False otherwise.
        """
        return False

    @property
    def transform_type(self) -> PerspectiveTransformationMethod:
        return PerspectiveTransformationMethod.AFFINE
