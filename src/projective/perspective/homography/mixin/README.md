# perspective.homography.mixin

## Overview

Concrete mixins wired into `HomographyMatrix`: homography estimation and projective map with dehomogenization.

## Components

| Component | Description |
|-----------|-------------|
| [factory.py](./factory.py) | `HomographyMatrixFactoryMixin` — `create_from_points` via `findHomography` |
| [transform.py](./transform.py) | `HomographyMatrixTransformMixin` — 3×3 map with divide-by-third coordinate |
