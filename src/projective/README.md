# projective

## Overview

Utilities for **projective geometry**: epipolar constraints at the package root (`EssentialMatrix`, `FundamentalMatrix`), and 2D **perspective transforms** (affine / homography) in the [`perspective/`](./perspective/README.md) subpackage.

## Components

| Component | Description |
|-----------|-------------|
| [perspective/README.md](./perspective/README.md) | Partial affine and homography matrix containers; point estimation returns `(matrix, mask)` |
| [essential_matrix.py](./essential_matrix.py) | Immutable `EssentialMatrix` (3×3) |
| [fundamental_matrix.py](./fundamental_matrix.py) | Immutable `FundamentalMatrix` (3×3) |
