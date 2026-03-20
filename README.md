# Projective

`projective` is a Python package for **projective geometry**: epipolar constraints (`EssentialMatrix`, `FundamentalMatrix`) at the package root, and 2D **perspective transforms** (affine / homography) under the perspective subpackage.

## Documentation

| Topic | Link |
|--------|------|
| Package overview and module index | [src/projective/README.md](src/projective/README.md) |
| Affine / homography API and math notes | [src/projective/perspective/README.md](src/projective/perspective/README.md) |

## Installation

From the project root:

```bash
pip install -e .
```

Or install dependencies only (use when developing without installing the package):

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from projective import (
    AffineMatrix,
    EssentialMatrix,
    FundamentalMatrix,
    HomographyMatrix,
)

# Epipolar geometry (3×3 matrices)
E = EssentialMatrix(value=np.eye(3))
F = FundamentalMatrix(value=np.eye(3))

# 2D transforms (OpenCV-style shapes: affine 2×3, homography 3×3)
affine = AffineMatrix(value=np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float64))
pts = np.array([[0.0, 0.0], [100.0, 50.0]])
warped = affine.projective_transformation(pts)

H = HomographyMatrix(value=np.eye(3))
out = H.projective_transformation(pts)
```

For estimation from point correspondences and registration helpers, see [src/projective/perspective/README.md](src/projective/perspective/README.md).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
