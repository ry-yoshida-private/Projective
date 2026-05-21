# perspective.affine.mixin

## Overview

Concrete mixins wired into `AffineMatrix`: partial affine estimation and affine map application to (N, 2) points.

## Components

| Component | Description |
|-----------|-------------|
| [factory.py](./factory.py) | `AffineMatrixFactoryMixin` — `create_from_points` via `estimateAffinePartial2D` |
| [transform.py](./transform.py) | `AffineMatrixTransformMixin` — 2×3 linear map on homogeneous points |
