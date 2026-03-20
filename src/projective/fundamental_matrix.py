from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class FundamentalMatrix:
    """
    FundamentalMatrix is a class that represents a fundamental matrix.

    Attributes:
    ----------
    value: np.ndarray
        The fundamental matrix with shape (3, 3).
    """
    value: np.ndarray

    def __post_init__(self):
        if self.value.shape != (3, 3):
            raise ValueError("Fundamental matrix must be a 3x3 matrix")

    @classmethod
    def from_points(
        cls, 
        points1: np.ndarray, 
        points2: np.ndarray,
        ransac_th: float = 3.0
        ) -> tuple[FundamentalMatrix, np.ndarray]:
        """
        Create a FundamentalMatrix from two sets of points.

        Parameters:
        ----------
        points1: np.ndarray
            Array of shape (N, 2) containing the coordinates of points in the first image.
        points2: np.ndarray
            Array of shape (N, 2) containing the coordinates of points in the second image.
        ransac_th: float
            RANSAC threshold.

        Returns:
        ----------
        tuple[FundamentalMatrix, np.ndarray]: The fundamental matrix and the mask.
        """
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        if points1.shape != points2.shape:
            raise ValueError(f"Points arrays must have the same shape, got {points1.shape} and {points2.shape}")
        if points1.ndim != 2:
            raise ValueError(f"Points arrays must be 2D, got {points1.ndim}D")
        if points1.shape[0] < 8:
            raise ValueError(f"At least 8 point pairs are required to estimate a fundamental matrix, got {points1.shape[0]}")
        if points1.shape[1] != 2:
            raise ValueError(f"Points arrays must have 2 columns, got {points1.shape[1]}")
        
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, ransac_th) # type: ignore
        if F is None:
            raise ValueError("Failed to estimate fundamental matrix")
        
        return cls(value=F), mask