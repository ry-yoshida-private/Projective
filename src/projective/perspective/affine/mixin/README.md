# perspective.affine.mixin

## Overview

Concrete mixins wired into `AffineMatrix`: partial affine estimation, matrix self-transformation, and 2D point projection.

## Components

| Component | Description |
|-----------|-------------|
| [factory.py](./factory.py) | `AffineFactoryMixin` — `create_from_points` via `estimateAffinePartial2D` |
| [transform.py](./transform.py) | `AffineTransformMixin` — 2×3 container views (6 / 1×6) via `super()`; `scale_correction` |
| [projective.py](./projective.py) | `AffineProjectiveMixin` — 2×3 linear map on homogeneous points |
