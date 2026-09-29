"""Training-fitted encodings of explicitly selected clinical features.

The paper prescribes one-hot categorical features and normalized continuous
features, but does not specify an imputer or scaler.  This implementation uses
training medians, population standard deviations, and explicit missing flags.
No field is selected automatically and no label is derived from a feature.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, Real):
        try:
            return math.isnan(float(value))
        except OverflowError:
            return False
    return False


def _numeric(value: object, field: str) -> float:
    if _is_missing(value):
        return float("nan")
    if isinstance(value, (Mapping, list, tuple, set, complex, np.ndarray)):
        raise ValueError(f"Numeric feature {field!r} contains a non-numeric value")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Numeric feature {field!r} contains a non-numeric value") from exc
    if not math.isfinite(number):
        raise ValueError(f"Numeric feature {field!r} must be finite or a missing value")
    return number


def _category(value: object, field: str) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, Real):
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError(f"Categorical feature {field!r} contains an unrepresentable number") from exc
        if math.isfinite(number):
            return str(number)
    raise ValueError(f"Categorical feature {field!r} must be a string, boolean, finite number, or missing")


def _field_names(fields: Sequence[str], kind: str) -> tuple[str, ...]:
    if isinstance(fields, str):
        raise ValueError(f"{kind} must be a sequence of field names, not a string")
    try:
        names = tuple(fields)
    except TypeError as exc:
        raise ValueError(f"{kind} must be a sequence of field names") from exc
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{kind} must contain nonempty string field names")
    if len(names) != len(set(names)):
        raise ValueError(f"{kind} contains duplicate field names")
    return names


def _rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    items = list(rows)
    if any(not isinstance(row, Mapping) for row in items):
        raise ValueError("Each structured row must be a mapping")
    return items


class StructuredEncoder:
    """Encode a fixed feature whitelist without learning from inference rows.

    Each numeric field produces its value followed by a missing indicator.
    Each categorical field produces sorted training categories followed by
    separate missing and unknown columns, even if training contains neither.
    ``standardize=False`` preserves the scale of observed continuous values.
    """

    def __init__(
        self,
        numeric_features: Sequence[str] = (),
        categorical_features: Sequence[str] = (),
        standardize: bool = True,
    ) -> None:
        self.numeric_features = _field_names(numeric_features, "numeric_features")
        self.categorical_features = _field_names(categorical_features, "categorical_features")
        if not self.numeric_features and not self.categorical_features:
            raise ValueError("At least one structured feature must be selected")
        if set(self.numeric_features) & set(self.categorical_features):
            raise ValueError("A field cannot be both numeric and categorical")
        if not isinstance(standardize, bool):
            raise ValueError("standardize must be a boolean")
        self.standardize = standardize
        self._numeric_stats: dict[str, dict[str, float]] = {}
        self._categories: dict[str, list[str]] = {}
        self._fitted = False

    def fit(self, rows: Sequence[Mapping[str, object]]) -> StructuredEncoder:
        items = _rows(rows)
        if not items:
            raise ValueError("Cannot fit a structured encoder on empty input")
        numeric_stats: dict[str, dict[str, float]] = {}
        for field in self.numeric_features:
            values = np.asarray([_numeric(row.get(field), field) for row in items], dtype=np.float64)
            observed = values[~np.isnan(values)]
            try:
                with np.errstate(over="raise", invalid="raise"):
                    median = float(np.median(observed)) if len(observed) else 0.0
                    filled = np.where(np.isnan(values), median, values)
                    mean = float(np.mean(filled))
                    scale = float(np.std(filled)) if np.any(filled != filled[0]) else 1.0
            except FloatingPointError as exc:
                raise ValueError(f"Numeric feature {field!r} is too large for stable statistics") from exc
            if scale == 0.0:
                scale = 1.0
            numeric_stats[field] = {"median": median, "mean": mean, "scale": scale}
        categories: dict[str, list[str]] = {}
        for field in self.categorical_features:
            values = {_category(row.get(field), field) for row in items}
            categories[field] = sorted(value for value in values if value is not None)
        self._numeric_stats = numeric_stats
        self._categories = categories
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("StructuredEncoder must be fitted before use")

    @property
    def feature_names_(self) -> list[str]:
        self._require_fitted()
        names: list[str] = []
        for field in self.numeric_features:
            key = json.dumps(field, ensure_ascii=False)
            names.extend([f"numeric[{key}]", f"numeric_missing[{key}]"])
        for field in self.categorical_features:
            key = json.dumps(field, ensure_ascii=False)
            names.extend(
                f"categorical[{key}]={json.dumps(value, ensure_ascii=False)}"
                for value in self._categories[field]
            )
            names.extend([f"categorical_missing[{key}]", f"categorical_unknown[{key}]"])
        return names

    def transform(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        self._require_fitted()
        items = _rows(rows)
        result = np.zeros((len(items), len(self.feature_names_)), dtype=np.float32)
        offset = 0
        for field in self.numeric_features:
            stats = self._numeric_stats[field]
            values = np.asarray([_numeric(row.get(field), field) for row in items], dtype=np.float64)
            missing = np.isnan(values)
            filled = np.where(missing, stats["median"], values)
            try:
                with np.errstate(over="raise", invalid="raise"):
                    if self.standardize:
                        filled = (filled - stats["mean"]) / stats["scale"]
                    if not np.all(np.isfinite(filled)) or np.any(np.abs(filled) > np.finfo(np.float32).max):
                        raise ValueError(f"Numeric feature {field!r} cannot be represented as finite float32")
                    result[:, offset] = filled.astype(np.float32)
            except FloatingPointError as exc:
                raise ValueError(f"Numeric feature {field!r} cannot be represented as finite float32") from exc
            result[:, offset + 1] = missing
            offset += 2
        for field in self.categorical_features:
            categories = self._categories[field]
            lookup = {value: index for index, value in enumerate(categories)}
            for index, row in enumerate(items):
                value = _category(row.get(field), field)
                column = len(categories) if value is None else lookup.get(value, len(categories) + 1)
                result[index, offset + column] = 1.0
            offset += len(categories) + 2
        return result

    def fit_transform(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        items = _rows(rows)
        return self.fit(items).transform(items)

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-compatible configuration and fitted state."""
        self._require_fitted()
        return {
            "version": 1,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "standardize": self.standardize,
            "numeric_stats": {field: dict(stats) for field, stats in self._numeric_stats.items()},
            "categories": {field: list(values) for field, values in self._categories.items()},
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> StructuredEncoder:
        """Restore and validate state; never refit from inference data."""
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise ValueError("Unsupported structured encoder state version")
        try:
            encoder = cls(state["numeric_features"], state["categorical_features"], state["standardize"])
            numeric_stats = state["numeric_stats"]
            categories = state["categories"]
            if not isinstance(numeric_stats, Mapping) or set(numeric_stats) != set(encoder.numeric_features):
                raise ValueError("Invalid numeric state fields")
            if not isinstance(categories, Mapping) or set(categories) != set(encoder.categorical_features):
                raise ValueError("Invalid categorical state fields")
            for field in encoder.numeric_features:
                stats = numeric_stats[field]
                if not isinstance(stats, Mapping) or set(stats) != {"median", "mean", "scale"}:
                    raise ValueError(f"Invalid numeric statistics for {field!r}")
                clean = {key: float(value) for key, value in stats.items()}
                if not all(math.isfinite(value) for value in clean.values()) or clean["scale"] <= 0:
                    raise ValueError(f"Invalid numeric statistics for {field!r}")
                encoder._numeric_stats[field] = clean
            for field in encoder.categorical_features:
                values = categories[field]
                if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                    raise ValueError(f"Invalid categories for {field!r}")
                if values != sorted(set(values)) or any(value != value.strip().lower() for value in values):
                    raise ValueError(f"Categories for {field!r} must be normalized, sorted, and unique")
                encoder._categories[field] = list(values)
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Invalid structured encoder state") from exc
        encoder._fitted = True
        return encoder


def encode_oct_status(value: object) -> float | None:
    """Encode the OCT status ordering in GlaBoost section III.E (page 3).

    The later ablation section describes a conflicting treatment of borderline
    values.  This helper implements only the explicit three-status definition;
    it is not an assertion that these fields exist in GRAPE.
    """
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        raise ValueError("OCT status must be a status string or missing")
    statuses = {"outside normal": 0.0, "borderline": 0.5, "within normal": 1.0}
    normalized = " ".join(value.strip().lower().split())
    if normalized not in statuses:
        raise ValueError(f"Unknown OCT status: {value!r}")
    return statuses[normalized]
