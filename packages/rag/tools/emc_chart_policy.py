"""Shared chart-axis and EMC quantity normalization policies."""

from __future__ import annotations

from typing import Iterable, Optional

from .emc_utils import normalize_unit


def frequency_extent(datasets: Iterable[dict]) -> dict:
    """Return positive-frequency union and common intersection in MHz."""
    dataset_extents = []
    for dataset in datasets:
        valid = [
            (float(start), float(end))
            for start, end in dataset.get("ranges", [])
            if start is not None and end is not None and start > 0 and end >= start
        ]
        if valid:
            dataset_extents.append((min(x[0] for x in valid), max(x[1] for x in valid)))
    if not dataset_extents:
        return {"union": None, "intersection": None}
    union = [min(x[0] for x in dataset_extents), max(x[1] for x in dataset_extents)]
    common_start = max(x[0] for x in dataset_extents)
    common_end = min(x[1] for x in dataset_extents)
    intersection = [common_start, common_end] if common_start <= common_end else None
    return {"union": union, "intersection": intersection}


def choose_frequency_axis(start: float, end: float) -> str:
    """Choose log for spans of at least two decades, otherwise linear."""
    if start <= 0 or end < start:
        return "value"
    return "log" if end / start >= 100 else "value"


def normalize_type_b_group(category: str, method: str, unit: str) -> tuple[str, str, str]:
    """Normalize a Type B method into a comparison-safe facet identity."""
    category_text = (category or "").strip().lower()
    method_text = (method or "").strip().lower()
    canonical_unit = _canonical_unit(unit)
    combined = f"{category_text} {method_text}"
    if "bci" in combined or "大电流" in combined:
        return "bci", "current_level", canonical_unit
    if "自由场" in combined or "free field" in combined or "anechoic" in combined:
        return "free_field", "electric_field", canonical_unit
    if "磁场" in combined or "magnetic" in combined:
        return "magnetic_field", "magnetic_field", canonical_unit
    if "传导" in combined or "conducted" in combined:
        return "conducted_immunity", _quantity_for_unit(canonical_unit), canonical_unit
    return "needs_review", _quantity_for_unit(canonical_unit), canonical_unit


def _canonical_unit(unit: Optional[str]) -> str:
    return normalize_unit(unit or "")


def _quantity_for_unit(unit: str) -> str:
    return {
        "V/m": "electric_field",
        "A/m": "magnetic_field",
        "mA": "current",
        "dBµA": "current_level",
    }.get(unit, "unknown")
