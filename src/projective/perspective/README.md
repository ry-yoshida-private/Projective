# perspective

## Overview

2D perspective transformation using **affine** (2×3) and **homography** (3×3) matrices: shared container API, optional estimation from corresponding points, and registration helpers that normalize raw arrays into the right matrix type.

## Mathematics

### Homogeneous coordinates

Image points use **homogeneous 2D** $(x, y, 1)^\top$. Affine: multiply by the 2×3 matrix. Homography: multiply by $H$, then **dehomogenize** (divide the first two components by the third).

### Affine map

OpenCV-style 2×3 $A$:

$$
\begin{bmatrix} x' \\ y' \end{bmatrix}
\sim
A
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
\sim
\begin{bmatrix}
a_{00} & a_{01} & a_{02} \\
a_{10} & a_{11} & a_{12}
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}.
$$

### Homography map

3×3 $H$ (defined up to a non-zero scalar):

$$
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix}
\sim
H
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
\sim
\begin{bmatrix}
h_{00} & h_{01} & h_{02} \\
h_{10} & h_{11} & h_{12} \\
h_{20} & h_{21} & h_{22}
\end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}.
$$

### Affine vs. homography

| Transformation | Affine (2×3) | Homography (3×3) | Geometric effect |
|----------------|--------------|------------------|------------------|
| Translation | ✅ | ✅ | Moving the origin $(t_x, t_y)$ |
| Rotation | ✅ | ✅ | Rotating around an axis |
| Uniform scaling | ✅ | ✅ | Changing the size (zoom in/out) |
| Shear (skew) | ✅ | ✅ | Tilting/slanting the image |
| Perspective | ❌ | ✅ | Foreshortening (vanishing points) |
| Parallelism | Preserved | Lost | Do parallel lines remain parallel? |
| Min. points | 3 | 4 | Points required for `create_from_points` |
| Degrees of freedom | 6 | 8 | Total independent variables |

## Components

### Module layout

| Component | Description |
|-----------|-------------|
| [method.py](./method.py) | Enum for transformation kind (Affine / Homography) and mapping to OpenCV motion types |
| [perspective_matrix.py](./perspective_matrix.py) | Abstract base class for Affine and Homography perspective transforms |
| [affine_matrix.py](./affine_matrix.py) | `AffineMatrix` container (shape 2×3) |
| [homography_matrix.py](./homography_matrix.py) | `HomographyMatrix` container (shape 3×3) |
| [register.py](./register.py) | Register an existing matrix or build one from point pairs |

## Example

Wrap a matrix with `register_perspective_matrix` (or estimate from point pairs with `register_perspective_matrix_from_points`), then map 2D points with `projective_transformation`.

```python
import numpy as np
from projective import (
    PerspectiveTransformationMethod,
    register_perspective_matrix,
    register_perspective_matrix_from_points,
)

# --- Homography: wrap an existing 3×3 ndarray ---
H = np.array(
    [[1.0, 0.1, 10.0], [0.0, 1.0, 5.0], [0.0002, 0.0, 1.0]],
    dtype=np.float64,
)
T = register_perspective_matrix(H, PerspectiveTransformationMethod.HOMOGRAPHY)
src = np.array([[0.0, 0.0], [100.0, 50.0]], dtype=np.float64)
dst = T.projective_transformation(src)  # shape (N, 2)

# --- Estimate from origin ↔ destination points (≥4 pairs for homography, ≥3 for affine) ---
origin = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
destination = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=np.float64)
T_fit = register_perspective_matrix_from_points(
    origin,
    destination,
    transform_type=PerspectiveTransformationMethod.HOMOGRAPHY,
)
mapped = T_fit.projective_transformation(src)

# --- Affine: pass a 2×3 matrix ---
A = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=np.float64)
T_affine = register_perspective_matrix(A, PerspectiveTransformationMethod.AFFINE)
dst_affine = T_affine.projective_transformation(src)
```
