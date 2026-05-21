import numpy as np
from typing import cast

from .matrix import PerspectiveMatrix
from .homography.matrix import HomographyMatrix
from .affine.matrix import AffineMatrix
from .method import PerspectiveTransformationMethod

def register_perspective_matrix(
    matrix: np.ndarray| AffineMatrix | HomographyMatrix | None,
    transform_type: PerspectiveTransformationMethod = PerspectiveTransformationMethod.HOMOGRAPHY
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
    PerspectiveMatrix:
        The registered perspective matrix.
    """

    if isinstance(matrix, (AffineMatrix, HomographyMatrix)):
        return matrix

    target_class = HomographyMatrix if transform_type == PerspectiveTransformationMethod.HOMOGRAPHY else AffineMatrix

    if matrix is None:
        return cast(PerspectiveMatrix, target_class.create_identity_matrix())
    return target_class(value=matrix)
