# perspective.mixin

## Overview

Abstract mixins that implement shared behavior on `PerspectiveMatrix`: OpenCV-based estimation from point pairs (`from_points`) and homogeneous 2D point projection.

## Components

| Component | Description |
|-----------|-------------|
| [factory.py](./factory.py) | `PerspectiveMatrixFactoryMixin` — point validation and `from_points` dispatch |
| [transform.py](./transform.py) | `PerspectiveMatrixTransformMixin` — `projective_transformation` for 2D points |
