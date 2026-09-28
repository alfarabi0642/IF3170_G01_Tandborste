# Core models and physics constraints
from .models import Package, Truck, ORIENTATIONS
from .constraints import (
    is_within_bounds,
    is_overlapping_3d,
    has_horizontal_overlap,
    is_supported,
    is_fragile_violated,
    is_capacity_exceeded,
    check_all_constraints,
)

__all__ = [
    "Package",
    "Truck",
    "ORIENTATIONS",
    "is_within_bounds",
    "is_overlapping_3d",
    "has_horizontal_overlap",
    "is_supported",
    "is_fragile_violated",
    "is_capacity_exceeded",
    "check_all_constraints",
]
