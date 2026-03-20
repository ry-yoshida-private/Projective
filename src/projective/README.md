# projective

## Overview

Utilities for **projective geometry**: epipolar constraints at the package root (`EssentialMatrix`, `FundamentalMatrix`), and 2D **perspective transforms** (affine / homography) in the [`perspective/`](./perspective/README.md) subpackage.

## Components

| Component | Description |
|-----------|-------------|
| [perspective/README.md](./perspective/README.md) | Affine and homography matrix containers, registration from arrays or point correspondences |
| [essential_matrix.py](./essential_matrix.py) | Immutable `EssentialMatrix` (3×3) |
| [fundamental_matrix.py](./fundamental_matrix.py) | `FundamentalMatrix` (3×3) |
