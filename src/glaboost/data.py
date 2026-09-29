"""Read-only input adapters; reference labels and visit metadata stay separate."""

import json
import math
import os
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


FUNDUS_NUMERIC_FEATURES = ("cup_to_disc_ratio",)
FUNDUS_CATEGORICAL_FEATURES = (
    "optic_disc_size",
    "isnt_rule_followed",
    "rim_pallor",
    "rim_color",
    "bayoneting",
    "sharp_edge",
    "laminar_dot_sign",
    "notching",
    "rim_thinning",
)

PathInput = Union[str, os.PathLike]


@dataclass
class VisitInput:
    """One visit's inputs; grouping/time metadata are not model features.

    ``image`` may hold original bytes, a path, or an already decoded image.
    ``human`` is a separate, explicitly enabled modality with independently
    documented provenance. Targets deliberately have no field in this object.
    """

    sample_id: str
    image: object = None
    structured: Mapping[str, object] = field(default_factory=dict)
    text: Optional[str] = None
    human: Mapping[str, object] = field(default_factory=dict)
    patient_id: Optional[str] = None
    eye_id: Optional[str] = None
    time_years: Optional[float] = None


@dataclass
class GrapeDataset:
    """All GRAPE visits and separate eye-level progression reference labels.

    The labels are progression outcomes, never glaucoma/normal diagnoses.
    Visits contain only raw CFP paths and contemporaneous IOP as predictors;
    VF measurements and baseline-only measurements are deliberately excluded.
    """

    visits: List[VisitInput]
    progression_labels: Dict[str, Dict[str, int]]

    def image_visits(self, min_visits: int = 3) -> List[VisitInput]:
        """Select eyes with at least ``min_visits`` available original CFPs.

        This is a cohort eligibility rule, not a predictive feature. Missing
        image visits are omitted; no follow-up visit is duplicated as baseline.
        """
        if isinstance(min_visits, bool) or not isinstance(min_visits, Integral) or min_visits < 1:
            raise ValueError("min_visits must be a positive integer")
        counts: Dict[str, int] = {}
        for visit in self.visits:
            if visit.image is not None:
                counts[visit.eye_id] = counts.get(visit.eye_id, 0) + 1
        return sorted(
            (v for v in self.visits if v.image is not None and counts[v.eye_id] >= min_visits),
            key=_visit_sort_key,
        )


def _visit_sort_key(visit: VisitInput):
    return (visit.patient_id or "", visit.eye_id or "", visit.time_years, visit.sample_id)


def _binary_label(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError(f"{context}: label must be an integer 0 or 1, got {value!r}")
    return int(value)


def _sample_id(value: object, seen: set, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: sample_id must be a nonempty string")
    if value in seen:
        raise ValueError(f"{context}: duplicate sample_id {value!r}")
    seen.add(value)
    return value


def _optional_string(value: object, field_name: str, context: str) -> Optional[str]:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{context}: {field_name} must be a string or null")
    return value


def _resolve_image_path(value: object, directory: Path, context: str) -> Optional[Path]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: image path must be a nonempty string or null")
    image_path = Path(value)
    if not image_path.is_absolute():
        image_path = directory / image_path
    return image_path.resolve()


def _grape_rows(sheet, required: Sequence[str]) -> List[Dict[str, object]]:
    """Flatten GRAPE's two header rows without depending on pandas inference."""
    rows = iter(sheet.iter_rows(values_only=True))
    first = next(rows, None)
    second = next(rows, None)
    if first is None or second is None:
        raise ValueError(f"{sheet.title}: expected two header rows")
    names = []
    parent = None
    for top, child in zip(first, second):
        if top is not None:
            parent = str(top).strip()
        name = parent
        if child is not None:
            name = f"{parent}/{child}"
        names.append(name)
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f"{sheet.title}: missing GRAPE columns {sorted(missing)}")
    for name in required:
        if names.count(name) != 1:
            raise ValueError(f"{sheet.title}: ambiguous duplicate column {name!r}")
    return [dict(zip(names, row)) for row in rows if any(value is not None for value in row)]


def _grape_eye(row: Mapping[str, object], context: str) -> Tuple[str, str]:
    patient = row["Subject Number"]
    if isinstance(patient, Integral) and not isinstance(patient, bool):
        patient = str(patient)
    elif isinstance(patient, str) and patient.strip():
        patient = patient.strip()
    else:
        raise ValueError(f"{context}: Subject Number must be an integer or nonempty string")
    laterality = row["Laterality"]
    if laterality not in ("OD", "OS"):
        raise ValueError(f"{context}: Laterality must be OD or OS")
    return patient, f"{patient}_{laterality}"


def load_grape(root: PathInput = "data/raw/grape") -> GrapeDataset:
    """Read GRAPE's retained Excel workbook and original CFP references.

    ``Follow-up`` already includes baseline visits, so only that sheet creates
    VisitInput records. ``Baseline`` supplies PLR2, PLR3 and MD-slope progression
    labels separately. The workbook's ``/`` image marker becomes ``None``.
    Available filenames must resolve to an existing original ``extracted/CFPs``
    file. Annotation overlays, VF, total visit counts and baseline-only RNFL
    measurements are never copied into model feature channels.
    """
    import openpyxl

    root = Path(root).resolve()
    workbook_path = root / "files" / "VF and clinical information.xlsx"
    image_directory = (root / "extracted" / "CFPs").resolve()
    outcome_columns = {
        "plr2": "Progression Status/PLR2",
        "plr3": "Progression Status/PLR3",
        "md_slope": "Progression Status/MD",
    }
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        if not {"Baseline", "Follow-up"}.issubset(workbook.sheetnames):
            raise ValueError("GRAPE workbook requires Baseline and Follow-up sheets")
        baseline = _grape_rows(
            workbook["Baseline"], ["Subject Number", "Laterality"] + list(outcome_columns.values())
        )
        followup = _grape_rows(
            workbook["Follow-up"],
            ["Subject Number", "Laterality", "Visit Number", "Interval Years", "IOP", "Corresponding CFP"],
        )
    finally:
        workbook.close()

    outcomes: Dict[str, Dict[str, int]] = {}
    for row_number, row in enumerate(baseline, start=3):
        context = f"Baseline row {row_number}"
        _, eye_id = _grape_eye(row, context)
        if eye_id in outcomes:
            raise ValueError(f"{context}: duplicate baseline eye_id {eye_id!r}")
        outcomes[eye_id] = {
            name: _binary_label(row[column], f"{context}, {column}")
            for name, column in outcome_columns.items()
        }

    visits: List[VisitInput] = []
    seen: set = set()
    visit_order: Dict[str, List[Tuple[int, float]]] = {}
    for row_number, row in enumerate(followup, start=3):
        context = f"Follow-up row {row_number}"
        patient_id, eye_id = _grape_eye(row, context)
        if eye_id not in outcomes:
            raise ValueError(f"{context}: eye_id {eye_id!r} has no baseline progression labels")
        number = row["Visit Number"]
        if isinstance(number, bool) or not isinstance(number, Integral) or number < 1:
            raise ValueError(f"{context}: Visit Number must be a positive integer")
        sample_id = _sample_id(f"{eye_id}_{number}", seen, context)
        elapsed = row["Interval Years"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, Real) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"{context}: Interval Years must be a finite nonnegative number")
        elapsed = float(elapsed)
        iop = row["IOP"]
        if iop is not None:
            if isinstance(iop, bool) or not isinstance(iop, Real) or not math.isfinite(iop):
                raise ValueError(f"{context}: IOP must be a finite number or null")
            iop = float(iop)
        filename = row["Corresponding CFP"]
        image = None
        if filename not in (None, "", "/"):
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(f"{context}: Corresponding CFP must be an original CFP filename")
            image = (image_directory / filename).resolve()
            if image.parent != image_directory:
                raise ValueError(f"{context}: Corresponding CFP must stay inside extracted/CFPs")
            if not image.is_file():
                raise FileNotFoundError(f"{context}: original CFP is missing: {image}")
        visits.append(VisitInput(
            sample_id=sample_id, image=image, structured={"iop": iop},
            patient_id=patient_id, eye_id=eye_id, time_years=elapsed,
        ))
        visit_order.setdefault(eye_id, []).append((int(number), elapsed))
    if not visits:
        raise ValueError("GRAPE workbook has no follow-up visit records")
    missing_visits = set(outcomes) - set(visit_order)
    if missing_visits:
        raise ValueError(f"Baseline eyes have no follow-up records: {sorted(missing_visits)}")
    for eye_id, entries in visit_order.items():
        entries.sort()
        if any(right[1] <= left[1] for left, right in zip(entries, entries[1:])):
            raise ValueError(f"{eye_id}: Interval Years must increase with Visit Number")
    return GrapeDataset(visits=sorted(visits, key=_visit_sort_key), progression_labels=outcomes)


def load_jsonl(
    path: PathInput, *, require_labels: bool = False
) -> Tuple[List[VisitInput], Optional[np.ndarray]]:
    """Load an explicit visit manifest with optional, separate binary targets.

    Permitted fields are ``sample_id``, ``image_path``, ``structured``, ``text``,
    ``human``, ``patient_id``, ``eye_id``, ``time_years`` and ``target``. Binary
    targets use 1 for the explicitly chosen positive endpoint and 0 otherwise;
    this reader does not assign a diagnostic or progression meaning to them.
    If any row supplies a target, every row must supply one. No target is ever
    inferred from the human/risk channel.
    Missing modalities are left missing for the model's configured validation.
    """
    path = Path(path).resolve()
    allowed = {
        "sample_id", "image_path", "structured", "text", "human",
        "patient_id", "eye_id", "time_years", "target",
    }
    visits: List[VisitInput] = []
    labels: List[int] = []
    seen: set = set()
    target_present: List[bool] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            context = f"{path.name}, line {line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{context}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{context}: expected a JSON object")
            unexpected = set(row) - allowed
            if unexpected:
                raise ValueError(f"{context}: unsupported manifest fields {sorted(unexpected)}")
            sample_id = _sample_id(row.get("sample_id"), seen, context)
            for field_name in ("structured", "human"):
                if not isinstance(row.get(field_name, {}), dict):
                    raise ValueError(f"{context}: {field_name} must be a JSON object")
            time_years = row.get("time_years")
            if time_years is not None:
                if isinstance(time_years, bool) or not isinstance(time_years, Real) or not math.isfinite(time_years):
                    raise ValueError(f"{context}: time_years must be a finite number or null")
                time_years = float(time_years)
            has_target = "target" in row
            if require_labels and not has_target:
                raise ValueError(f"{context}: target is required")
            if has_target:
                labels.append(_binary_label(row["target"], context))
            target_present.append(has_target)
            visits.append(
                VisitInput(
                    sample_id=sample_id,
                    image=_resolve_image_path(row.get("image_path"), path.parent, context),
                    structured=dict(row.get("structured", {})),
                    text=_optional_string(row.get("text"), "text", context),
                    human=dict(row.get("human", {})),
                    patient_id=_optional_string(row.get("patient_id"), "patient_id", context),
                    eye_id=_optional_string(row.get("eye_id"), "eye_id", context),
                    time_years=time_years,
                )
            )
    if not visits:
        raise ValueError(f"{path.name}: no visit records")
    if any(target_present) and not all(target_present):
        raise ValueError(f"{path.name}: targets must be provided for every record or none")
    return visits, np.asarray(labels, dtype=np.int64) if labels else None
