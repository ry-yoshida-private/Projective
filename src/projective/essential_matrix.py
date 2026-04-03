from __future__ import annotations

import cv2
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

    @classmethod
    def from_points(
        cls,
        points1: np.ndarray,
        points2: np.ndarray,
        intrinsic_matrix: np.ndarray,
        prob: float = 0.999,
        threshold: float = 1.0
        ) -> tuple[EssentialMatrix, np.ndarray]:
        """
        Create an essential matrix from a set of points and an intrinsic matrix.

        Parameters
        ----------
        points1: np.ndarray
            Points in the first image.
        points2: np.ndarray
            Points in the second image.
        intrinsic_matrix: np.ndarray
            Intrinsic matrix.
        prob: float
            Probability of the essential matrix.
        threshold: float
            Threshold of the essential matrix.

        Returns
        -------
        tuple[EssentialMatrix, np.ndarray]:
            - EssentialMatrix: The estimated essential matrix.
            - np.ndarray: The mask of the estimated essential matrix.
        """
        if points1.shape != points2.shape:
            raise ValueError("Points1 and points2 must have the same shape")
        if points1.shape[0] < 5:
            raise ValueError("At least 5 points are required to estimate an essential matrix")
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be a 3x3 matrix")

        E, e_mask = cv2.findEssentialMat(
            points1, 
            points2, 
            intrinsic_matrix, 
            method=cv2.RANSAC, 
            prob=prob, 
            threshold=threshold
            )
        return cls(value=E), e_mask

