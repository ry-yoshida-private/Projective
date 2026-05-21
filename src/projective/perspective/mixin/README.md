# perspective.mixin

## Overview

Abstract mixins for `PerspectiveMatrix`: estimation from point pairs, matrix self-transformation, and 2D point projection.

## Components

| Component | Description |
|-----------|-------------|
| [factory.py](./factory.py) | `PerspectiveFactoryMixin` — point validation and `from_points` dispatch |
| [transform.py](./transform.py) | `PerspectiveTransformMixin` — generic container views and `scale_correction` |
| [projective.py](./projective.py) | `PerspectiveProjectiveMixin` — `projective_transformation` for 2D points |
