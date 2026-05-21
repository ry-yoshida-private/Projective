# perspective.homography.mixin

## Overview

Concrete mixins wired into `HomographyMatrix`: homography estimation, matrix self-transformation, and 2D point projection.

## Components

| Component | Description |
|-----------|-------------|
| [factory.py](./factory.py) | `HomographyFactoryMixin` — `create_from_points` via `findHomography` |
| [transform.py](./transform.py) | `HomographyTransformMixin` — 3×3 container views (9 / 1×9) via `super()`; inverse, transpose, `scale_correction` |
| [projective.py](./projective.py) | `HomographyProjectiveMixin` — 3×3 map with divide-by-third coordinate |
