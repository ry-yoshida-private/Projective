import numpy as np

from .perspective_matrix import PerspectiveMatrix
from .homography_matrix import HomographyMatrix
from .affine_matrix import AffineMatrix
from .method import PerspectiveTransformationMethod

def register_perspective_matrix(
    matrix: np.ndarray| AffineMatrix| HomographyMatrix | None,  
    transform_type: PerspectiveTransformationMethod
    ) -> PerspectiveMatrix:
    """
    Register a perspective matrix.
    
    Parameters
    ----------
    matrix: np.ndarray| AffineMatrix| HomographyMatrix | None
        The matrix to register.
        If the matrix is None, the identity matrix will be returned.
    transform_type: PerspectiveTransformationMethod
        The type of the transformation.

    Returns
    -------
    PerspectiveMatrixContainer:
        The registered perspective matrix.
    """

    if isinstance(matrix, (AffineMatrix, HomographyMatrix)):
        return matrix

    target_class = HomographyMatrix if transform_type == PerspectiveTransformationMethod.HOMOGRAPHY else AffineMatrix

    if matrix is None:
        return target_class.create_identity_matrix()
    if isinstance(matrix, np.ndarray):
        return target_class(value=matrix)
    raise TypeError(f"Unsupported matrix type: {type(matrix)}")

def register_perspective_matrix_from_points(
    origin_points: np.ndarray, 
    destination_points: np.ndarray, 
    ransac_th: float = 3.0, 
    transform_type: PerspectiveTransformationMethod = PerspectiveTransformationMethod.HOMOGRAPHY
    ) -> PerspectiveMatrix:
    """
    Register a perspective matrix from a set of origin and destination points.

    Parameters:
    ----------
    origin_points: np.ndarray
        Origin points in homogeneous coordinates (n, 2).
    destination_points: np.ndarray
        Destination points in homogeneous coordinates (n, 2).
    ransac_th: float
        RANSAC threshold.
    transform_type: PerspectiveTransformationMethod
        The type of the transformation.

    Returns
    -------
    PerspectiveMatrixContainer:
        The registered perspective matrix.
    """
    match transform_type:
        case PerspectiveTransformationMethod.AFFINE:
            return AffineMatrix.create_from_points(
                origin_points=origin_points, 
                destination_points=destination_points, 
                ransac_th=ransac_th
                )
        case PerspectiveTransformationMethod.HOMOGRAPHY:
            return HomographyMatrix.create_from_points(
                origin_points=origin_points, 
                destination_points=destination_points, 
                ransac_th=ransac_th
                )
