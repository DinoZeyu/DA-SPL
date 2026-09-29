"""Cohort-level, auditable reports from completed longitudinal evaluations.

No fitting, clinical interpretation, or result fabrication happens here. HTML
embeds its figures so the report can be shared without its companion directory.
"""

import base64
from collections import Counter
import csv
import html
import json
import math
from pathlib import Path


ENDPOINTS = ("plr2", "plr3", "md_slope")
ENDPOINT_NAMES = {"plr2": "PLR2", "plr3": "PLR3", "md_slope": "MD slope"}
METRICS = ("balanced_accuracy", "auroc", "auprc", "sensitivity", "specificity", "f1")
METHOD_NAMES = {"latest": "Latest visit", "longitudinal": "Longitudinal integration"}
BLUE, ORANGE = "#0072B2", "#E69F00"


def _clean(value):
    """Serialize unavailable numeric results as JSON null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _number(value, digits=3, signed=False):
    if value is None:
        return "not estimable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "not estimable"
    return format(number, "+." + str(digits) + "f" if signed else "." + str(digits) + "f")


def _interval(metric, signed=False):
    metric = metric or {}
    point = _number(metric.get("estimate"), signed=signed)
    lo, hi = metric.get("ci_low"), metric.get("ci_high")
    if lo is None or hi is None:
        return point + " (95% CI not estimable)"
    return point + " (95% CI " + _number(lo, signed=signed) + " to " + _number(hi, signed=signed) + ")"


def _count(value):
    return "not recorded" if value is None else str(value)


def _distribution(value):
    value = value or {}
    return "{} (IQR {} to {})".format(
        _number(value.get("median"), 1), _number(value.get("q1"), 1), _number(value.get("q3"), 1)
    )


def _summary(evaluation, cohort, provenance, synthetic):
    internal = provenance.get("validation_design") == "internal_nested_patient_cv"
    external = provenance.get("validation_design") == "external_fixed_detector"
    opening = (
        "We conducted a retrospective internal validation of a paper-based GlaBoost architecture adapted "
        "to progression assessment in the GRAPE longitudinal glaucoma cohort. The image-based visit "
        "models and temporal mappings were evaluated together using nested patient-grouped cross-validation. "
        if internal else
        "We conducted a retrospective external evaluation of a fixed visit-level detector on the GRAPE cohort, "
        "with internally cross-validated progression mappings. External independence is based on the supplied "
        "detector provenance declaration. "
        if external else
        "We conducted a retrospective evaluation on the GRAPE longitudinal glaucoma cohort. "
        "External validation is not established because independence of the visit-level detector from GRAPE "
        "has not been documented. "
    )
    if synthetic:
        opening = "SYNTHETIC TEST DATA — SOFTWARE CHECK ONLY. These numbers are not study findings. " + opening
    source = cohort.get("source", {})
    text = opening + (
        "Of {} source eyes, {} eyes from {} patients met the prespecified eligibility criteria, providing {} "
        "evaluable visits, a median of {} visits per eye over {} months between the first and last evaluable "
        "visits. Progression was evaluated separately using GRAPE's PLR2, PLR3, and MD-slope labels. "
    ).format(
        _count(source.get("n_eyes")), _count(cohort.get("n_eyes")), _count(cohort.get("n_patients")),
        _count(cohort.get("n_visits")), _number(cohort.get("visits_per_eye", {}).get("median"), 1),
        _number(cohort.get("followup_months", {}).get("median"), 1),
    )
    text += (
        "A separate visit model was trained for each endpoint and outer fold, using only training patients. "
        "Within each fold, the same trained model was applied independently to each held-out visit for both "
        "comparators; its outputs were integrated using prespecified temporal changes and trends. "
        "Training visit labels were inherited from each eye's retrospective progression outcome; the resulting "
        "scores are endpoint-specific evidence, not labels of disease state at an individual visit. "
        if internal else
        "The fixed detector was applied independently at each eligible visit; its outputs were integrated "
        "using prespecified temporal changes and trends. "
    )
    deltas = []
    for name in ENDPOINTS:
        result = evaluation.get("endpoints", {}).get(name, {})
        if result.get("status") != "ok":
            text += "{} was not estimable: {}. ".format(ENDPOINT_NAMES[name], result.get("reason", "no valid evaluation"))
            continue
        metrics = result.get("metrics", {})
        latest = metrics.get("latest", {}).get("balanced_accuracy", {})
        longitudinal = metrics.get("longitudinal", {}).get("balanced_accuracy", {})
        delta = result.get("delta_balanced_accuracy", {})
        text += (
            "For {}, balanced accuracy changed from {} with the latest visit to {} with longitudinal "
            "integration (paired difference {}). "
        ).format(ENDPOINT_NAMES[name], _interval(latest), _interval(longitudinal), _interval(delta, signed=True))
        lo, hi = delta.get("ci_low"), delta.get("ci_high")
        if lo is None or hi is None:
            text += "The paired interval was not estimable, so its uncertainty cannot be assessed. "
        elif lo > 0:
            text += "The paired interval favored higher balanced accuracy with longitudinal integration. "
        elif hi < 0:
            text += "The paired interval favored lower balanced accuracy with longitudinal integration. "
        else:
            text += "The paired interval included zero; the direction of the difference remains uncertain. "
        deltas.append(delta)
    if len(deltas) == 3 and all(d.get("estimate") is not None and d["estimate"] > 0 for d in deltas):
        text += "Point estimates favored longitudinal integration across all three definitions. "
    if deltas and all(d.get("ci_low") is not None and d["ci_low"] > 0 for d in deltas) and len(deltas) == 3:
        text += (
            "The conditional paired intervals were above zero for all three endpoints; this supports higher "
            "balanced accuracy within this analysis, without establishing prospective or causal benefit. "
        )
    else:
        text += "The results do not establish a consistent gain across all three progression definitions. "
    return text + (
        "The confidence intervals are patient-cluster bootstrap intervals conditional on the fitted out-of-fold "
        "predictions; they do not include uncertainty from refitting the cross-validation pipeline. "
        "This assessment concerns progression during the observed follow-up period, not prediction of future progression."
    )


def _primary_rows(evaluation):
    rows = []
    for endpoint in ENDPOINTS:
        result = evaluation.get("endpoints", {}).get(endpoint, {})
        row = {"endpoint": endpoint, "status": result.get("status", "not_estimable"),
               "reason": result.get("reason", ""), "n_eyes": result.get("n_eyes"),
               "n_patients": result.get("n_patients"), "n_positive_eyes": result.get("n_positive_eyes"),
               "n_positive_patients": result.get("n_positive_patients"), "n_splits": result.get("n_splits")}
        for method in METHOD_NAMES:
            metric = result.get("metrics", {}).get(method, {}).get("balanced_accuracy", {})
            for key in ("estimate", "ci_low", "ci_high"):
                row[method + "_" + key] = metric.get(key)
        for key in ("estimate", "ci_low", "ci_high"):
            row["delta_" + key] = result.get("delta_balanced_accuracy", {}).get(key)
        row.update({"bootstrap_" + key: value for key, value in result.get("bootstrap", {}).items()})
        rows.append(row)
    return rows


def _supplementary_rows(evaluation):
    rows = []
    for endpoint in ENDPOINTS:
        result = evaluation.get("endpoints", {}).get(endpoint, {})
        for method in METHOD_NAMES:
            for name in METRICS:
                metric = result.get("metrics", {}).get(method, {}).get(name, {})
                rows.append({"endpoint": endpoint, "method": method, "metric": name,
                             "estimate": metric.get("estimate"), "ci_low": metric.get("ci_low"),
                             "ci_high": metric.get("ci_high"), "status": result.get("status", "not_estimable")})
    return rows


def _write_csv(path, rows, defaults):
    fields = list(defaults)
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def _style():
    return {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
            "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
            "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False}


def _draw_estimate(ax, metric, y, color, marker):
    point = metric.get("estimate")
    if point is None or not math.isfinite(float(point)):
        return
    lo, hi = metric.get("ci_low"), metric.get("ci_high")
    if lo is not None and hi is not None:
        ax.hlines(y, lo, hi, color=color, linewidth=1.4)
        ax.vlines([lo, hi], y - .035, y + .035, color=color, linewidth=1.2)
    ax.plot(point, y, marker=marker, color=color, markersize=6, linestyle="none", clip_on=False)


def _primary_figure(evaluation, synthetic=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), gridspec_kw={"width_ratios": [1.55, 1]})
        fig.subplots_adjust(left=.19, right=.975, bottom=.25, top=.76, wspace=.22)
        ax, delta_ax = axes
        ax.set_xlim(0, 1)
        delta_ax.set_xlim(-1, 1)
        ax.set_xticks([0, .25, .5, .75, 1])
        delta_ax.set_xticks([-1, -.5, 0, .5, 1])
        labels = []
        for row, endpoint in enumerate(ENDPOINTS):
            y = 2 - row
            result = evaluation.get("endpoints", {}).get(endpoint, {})
            labels.append("{}\n{} eyes / {} patients".format(
                ENDPOINT_NAMES[endpoint], _count(result.get("n_eyes")), _count(result.get("n_patients"))))
            if result.get("status") != "ok":
                ax.text(.5, y, "Not estimable", ha="center", va="center", color="#555555", fontsize=8)
                delta_ax.text(0, y, "Not estimable", ha="center", va="center", color="#555555", fontsize=8)
                continue
            for method, offset, color, marker in (("latest", .10, BLUE, "o"), ("longitudinal", -.10, ORANGE, "s")):
                metric = result.get("metrics", {}).get(method, {}).get("balanced_accuracy", {})
                _draw_estimate(ax, metric, y + offset, color, marker)
            delta = result.get("delta_balanced_accuracy", {})
            _draw_estimate(delta_ax, delta, y + .04, "#333333", "D")
            text = "{} [{} to {}]".format(_number(delta.get("estimate"), signed=True),
                  _number(delta.get("ci_low"), signed=True), _number(delta.get("ci_high"), signed=True))
            if delta.get("ci_low") is None or delta.get("ci_high") is None:
                text = "{} (CI not estimable)".format(_number(delta.get("estimate"), signed=True))
            delta_ax.text(0, y - .24, text, ha="center", va="center", fontsize=7)
        for current in axes:
            current.set_ylim(-.5, 2.5)
            current.grid(axis="x", color="#E5E5E5", linewidth=.6)
            current.set_axisbelow(True)
            current.tick_params(axis="y", length=0)
        ax.set_yticks([2, 1, 0], labels)
        delta_ax.set_yticks([])
        ax.set_xlabel("Balanced accuracy")
        delta_ax.set_xlabel("Paired difference (B - A)")
        delta_ax.axvline(0, color="#777777", linestyle="--", linewidth=.8)
        ax.set_title("a  Latest visit vs longitudinal", loc="left", pad=14)
        delta_ax.set_title("b  Paired difference", loc="left", pad=14)
        fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", linestyle="none", label="A: latest visit"),
                            Line2D([], [], color=ORANGE, marker="s", linestyle="none", label="B: longitudinal integration")],
                   loc="upper center", bbox_to_anchor=(.56, .925), ncol=2, frameon=False)
        fig.suptitle("SYNTHETIC TEST DATA — NOT STUDY RESULTS" if synthetic else "GRAPE: paired progression assessment", y=.99, fontsize=11)
        fig.text(.02, .115, "Points: pooled out-of-fold estimates; bars: 95% patient-cluster percentile bootstrap intervals.", fontsize=8)
        fig.text(.02, .066, "Intervals condition on fitted predictions; no model refitting. Endpoints evaluated separately; no p-values.", fontsize=8)
        return fig


def _selected_eyes(eye_records, internal=False):
    ordered = sorted(eye_records, key=lambda row: (str(row.get("patient_id", "")), str(row.get("eye_id", ""))))
    if internal:
        return [row for endpoint in ENDPOINTS for row in
                [record for record in ordered if record.get("endpoint") == endpoint][:2]]
    return ordered[:6]


def _trajectory_figure(eye_records, synthetic=False, internal=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    selected = _selected_eyes(eye_records, internal=internal)
    slots = {}
    if internal:
        for column, endpoint in enumerate(ENDPOINTS):
            for row, record in enumerate(record for record in selected if record["endpoint"] == endpoint):
                slots[row * 3 + column] = record
    else:
        slots = dict(enumerate(selected))
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.5), sharex=True, sharey=True)
        fig.subplots_adjust(left=.11, right=.98, bottom=.28, top=.82, wspace=.18, hspace=.55)
        max_time = max((max(row.get("times", [0]) or [0]) for row in selected), default=1)
        for index, ax in enumerate(axes.flat):
            if index not in slots:
                ax.set_visible(False)
                continue
            record = slots[index]
            times, scores = record.get("times", []), record.get("scores", [])
            if len(times) != len(scores):
                raise ValueError("Trajectory times and scores must have equal lengths")
            pairs = sorted(zip(times, scores))
            if pairs:
                ax.plot([v[0] for v in pairs], [v[1] for v in pairs], color=BLUE, marker="o", markersize=3.5, linewidth=1, clip_on=False)
            title = "{}  {}".format(chr(97 + index), record.get("eye_id", "eye"))
            if internal:
                title = "{}  {} / {}\nOuter fold {}".format(
                    chr(97 + index), ENDPOINT_NAMES[record["endpoint"]], record.get("eye_id", "eye"),
                    record.get("outer_fold", record.get("fold", "not recorded")))
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, max(1, max_time) * 1.03)
            ax.set_yticks([0, .5, 1])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
            ax.grid(color="#E5E5E5", linewidth=.6)
            if index // 3 == 1:
                ax.set_xlabel("Time from baseline (years)", fontsize=8)
            if index % 3 == 0 and not internal:
                ax.set_ylabel("Visit-level diagnosis score")
        if internal:
            fig.text(.025, .55, "Visit-level progression evidence", rotation="vertical", va="center", fontsize=9)
        if not selected:
            fig.text(.5, .5, "No eligible score trajectories", ha="center", va="center")
        fig.suptitle("SYNTHETIC TEST DATA — NOT STUDY RESULTS" if synthetic else "Illustrative visit-level score histories", y=.98, fontsize=11)
        fig.text(.02, .15, "First two held-out eyes per endpoint, sorted by patient and eye ID; no outcome-based selection."
                 if internal else "First six eligible eyes sorted by patient and eye ID; selection does not use outcomes or performance.", fontsize=8)
        if internal:
            fig.text(.02, .105, "Models differ by endpoint and outer fold; each sequence uses one model fixed across its visits.", fontsize=8)
        fig.text(.02, .060, "Points are observed visit scores; lines guide the eye. Score changes alone do not establish progression.", fontsize=8)
        return fig


def _save_figures(run_dir, evaluation, eye_records, synthetic, internal=False):
    import matplotlib.pyplot as plt

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    for name, builder, values in (("primary_comparison", _primary_figure, evaluation),
                                   ("score_trajectories", _trajectory_figure, eye_records)):
        fig = builder(values, synthetic=synthetic, **({"internal": internal} if name == "score_trajectories" else {}))
        try:
            # Font embedding is consulted at export time, after builders return.
            with plt.rc_context(_style()):
                for extension in ("png", "pdf", "svg"):
                    with (figures / (name + "." + extension)).open("xb") as stream:
                        fig.savefig(stream, format=extension, dpi=300)
        finally:
            plt.close(fig)


def _table(headers, rows):
    return ("table", headers, rows)


def _compact_provenance(value):
    """Keep thousands of feature names out of the professor-facing narrative."""
    if isinstance(value, dict):
        if len(value) > 20:
            return {"entry_count": len(value), "full_values": "See provenance.json"}
        return {key: _compact_provenance(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > 20:
            return {"item_count": len(value), "full_values": "See provenance.json"}
        return [_compact_provenance(item) for item in value]
    return value


def _report_blocks(evaluation, cohort, provenance, eye_records, synthetic):
    internal = provenance.get("validation_design") == "internal_nested_patient_cv"
    external = provenance.get("validation_design") == "external_fixed_detector"
    title = (
        "Retrospective internal validation of longitudinal glaucoma progression assessment on GRAPE" if internal else
        "Preliminary external evaluation of longitudinal glaucoma progression assessment" if external else
        "Preliminary retrospective assessment of longitudinal glaucoma progression")
    if synthetic:
        title = "SYNTHETIC TEST REPORT — " + title
    blocks = [("h1", title)]
    if synthetic:
        blocks.append(("p", "SYNTHETIC TEST DATA. This document checks the software and must not be presented as research findings."))
    blocks.append(("p", "Run: {}. Created: {}.".format(provenance.get("run_name", "not recorded"), provenance.get("created_at", "not recorded"))))
    if not external and not internal:
        blocks.append(("p", "PROVISIONAL: external validation is not established. The detector's independence from GRAPE requires documented training and model-selection provenance."))
    blocks.extend([("h2", "Cohort-level summary"), ("p", _summary(evaluation, cohort, provenance, synthetic))])
    blocks.append(("h2", "Eligible cohort"))
    blocks.append(_table(["Quantity", "Value"], [
        ["Patients / eyes / evaluable visits", "{} / {} / {}".format(_count(cohort.get("n_patients")), _count(cohort.get("n_eyes")), _count(cohort.get("n_visits")))],
        ["Evaluable visits per eye, median (IQR)", _distribution(cohort.get("visits_per_eye"))],
        ["First-to-last evaluable visit, months, median (IQR)", _distribution(cohort.get("followup_months"))],
        ["Source patients / eyes / visits / visits with CFP", " / ".join(_count(cohort.get("source", {}).get(k)) for k in ("n_patients", "n_eyes", "n_visits", "n_visits_with_cfp"))],
        ["Excluded eyes", str(len(cohort.get("exclusions", [])))],
    ]))
    prevalence = []
    for ep in ENDPOINTS:
        counts = cohort.get("progression", {}).get(ep, {})
        prevalence.append([ENDPOINT_NAMES[ep], _count(counts.get("positive_eyes")), _count(counts.get("total_eyes")), _number(counts.get("prevalence"))])
    blocks.append(_table(["Progression definition", "Positive eyes", "Evaluable eyes", "Prevalence"], prevalence))
    modalities = cohort.get("modalities", {})
    iop = modalities.get("contemporaneous_iop", {})
    cfp = modalities.get("original_cfp", {})
    source = cohort.get("source", {})
    blocks.append(_table(["GRAPE field / modality", "Availability", "Role in this analysis"], [
        ["Original CFP / Corresponding CFP", "{} of {} source visits; {} of {} included visits".format(
            _count(source.get("n_visits_with_cfp")), _count(source.get("n_visits")),
            _count(cfp.get("available_visits", cohort.get("n_visits"))), _count(cohort.get("n_visits"))),
         "Visit-model input; original photographs only, no annotation overlays" if internal else
         "Fixed detector input; original photographs only, no annotation overlays"],
        ["Contemporaneous IOP", "{} available; {} missing in included visits".format(
            _count(iop.get("available_visits")), _count(iop.get("missing_visits"))),
         "Not used in the image-only progression model" if internal else
         "Enabled in fixed detector" if iop.get("selected") else "Not enabled in fixed detector"],
        ["Clinical text / human risk assessment", "Unavailable as longitudinal GRAPE inputs", "Not used; not synthesized"],
        ["Baseline OCT / RNFL", "Baseline-only measurements", "Not copied across subsequent visits; not used"],
        ["Visual field / PLR2, PLR3, MD-slope labels", "Released progression reference labels",
         "Eye-level supervision for training patients and reference outcomes for held-out patients; never input features"
         if internal else "Outcomes only; excluded from predictors"],
        ["Subject Number / Laterality / Interval Years", "Patient, eye, actual elapsed years", "Grouping and timing metadata; not visit-detector predictors"],
    ]))
    window = cohort.get("observation_window", {})
    if window:
        blocks.append(("p", "In {} eligible eyes, the last available CFP preceded the last recorded visit. "
                       "The CFP-to-last-recorded-visit gap was {} months (median and IQR as supplied). "
                       "The follow-up duration reported above is the eligible CFP observation span. Released "
                       "GRAPE reference labels use their original ascertainment period and are not recomputed "
                       "at the final CFP; these windows may differ. This does not define a prospective prediction "
                       "horizon.".format(_count(window.get("eyes_last_cfp_before_last_recorded_visit")),
                                        _distribution(window.get("cfp_to_last_recorded_visit_months")))))
    blocks.extend([("h2", "Primary paired comparison"), ("p", "Balanced accuracy and differences are on the 0–1 scale; a difference of 0.01 equals one percentage point. All intervals below are 95% patient-cluster percentile bootstrap intervals.")])
    primary = []
    for ep in ENDPOINTS:
        result = evaluation.get("endpoints", {}).get(ep, {})
        if result.get("status") != "ok":
            primary.append([ENDPOINT_NAMES[ep], "Not estimable", "Not estimable", result.get("reason", "No valid evaluation")])
        else:
            primary.append([ENDPOINT_NAMES[ep], _interval(result["metrics"]["latest"]["balanced_accuracy"]),
                            _interval(result["metrics"]["longitudinal"]["balanced_accuracy"]), _interval(result.get("delta_balanced_accuracy"), signed=True)])
    blocks.append(_table(["Endpoint", "A: latest visit", "B: longitudinal", "Paired difference B − A"], primary))
    blocks.append(("image", "figures/primary_comparison.png", "Paired comparison: pooled out-of-fold balanced accuracy and conditional 95% confidence intervals. Each endpoint uses the same eyes, visit scores, and held-out patients for A and B."))
    blocks.append(("h2", "Additional performance measures"))
    supplemental = []
    for ep in ENDPOINTS:
        result = evaluation.get("endpoints", {}).get(ep, {})
        if result.get("status") != "ok":
            continue
        for metric in METRICS[1:]:
            supplemental.append([ENDPOINT_NAMES[ep], "Average precision (AUPRC)" if metric == "auprc" else metric.upper() if metric in ("auroc", "f1") else metric.title(),
                                 _interval(result["metrics"].get("latest", {}).get(metric)),
                                 _interval(result["metrics"].get("longitudinal", {}).get(metric))])
    blocks.append(_table(["Endpoint", "Metric", "A: latest visit", "B: longitudinal"], supplemental))
    blocks.append(("h2", "Descriptive interpretation of the score histories"))
    blocks.append(("image", "figures/score_trajectories.png",
                   "Up to two held-out histories per endpoint, selected by sorted patient and eye ID without outcome "
                   "or performance selection. Models differ by endpoint and outer fold; each sequence uses one "
                   "model fixed across its visits. The plots show model evidence, not anatomical explanations."
                   if internal else
                   "Illustrative histories selected by sorted patient and eye ID, without outcome or performance selection. "
                   "No diagnostic interpretation of fundus anatomy is inferred from these plots."))
    selected = _selected_eyes(eye_records, internal=internal)
    trajectory_rows = []
    for eye in selected:
        scores, times = eye.get("scores", []), eye.get("times", [])
        prefix = ([ENDPOINT_NAMES[eye["endpoint"]], str(eye.get("outer_fold", eye.get("fold", "not recorded")))]
                  if internal else [])
        trajectory_rows.append(prefix + [str(eye.get("eye_id", "")), str(len(scores)),
                                _number(max(times) - min(times), 2) if times else "not recorded",
                                _number(max(scores) - min(scores)) if scores else "not recorded"])
    blocks.append(_table((["Endpoint", "Outer fold"] if internal else []) +
                         ["Illustrative eye", "Visits", "Observed span (years)", "Score range (maximum − minimum)"], trajectory_rows))
    blocks.append(("p",
                   "A nearly flat trajectory indicates little change in the learned visit evidence; a changing "
                   "trajectory can reflect image differences or acquisition variability. The eye-level training "
                   "outcome does not establish when progression began or whether it was present at an individual "
                   "visit. Scores are neither disease-severity measurements nor prospectively calibrated risks. "
                   "Differences across endpoints or outer folds also reflect different fitted models. These examples "
                   "illustrate inputs to temporal aggregation and do not replace cohort-level reference-label evaluation."
                   if internal else
                   "A nearly flat trajectory indicates little change in the detector's score, which may reflect stable appearance or limited sensitivity to progression. A changing trajectory can also reflect image acquisition variability. Neither pattern establishes clinical stability or deterioration. Diagnosis scores are not validated severity measurements or future progression risks. These examples illustrate inputs to the temporal summary, and do not replace cohort-level reference-label evaluation."))
    blocks.append(("h2", "Prespecified analysis and reproducibility"))
    config = evaluation.get("config", {})
    gpu_statistics = config.get("logistic_solver") == "torch_newton"
    tree_method = config.get("base_model_config", {}).get("tree_method", "hist")
    tree_description = ("GPU histogram tree building (gpu_hist), GPU prediction and the recorded seed"
                        if tree_method == "gpu_hist" else
                        "CPU histogram tree building and the recorded seed")
    blocks.append(("p", "Eligibility requires at least {} visits per eye with an available original CFP and a valid "
                   "visit score, ordered by actual elapsed years. Visits without a CFP are omitted; source cohort "
                   "visit counts are not treated as available image counts.".format(
                       _count(provenance.get("minimum_cfp_visits_per_eye")))))
    if internal:
        blocks.append(("p", "Each eye is a longitudinal unit. All visits and both eyes from a patient remain together "
                       "in both outer and inner folds. Only original CFP images enter the visit model. Visual-field "
                       "measurements, progression labels, patient IDs, elapsed times, and visit counts are never visit-model "
                       "input features. Original files are read without modification, and baseline-only measurements "
                       "are not copied to later visits. Frozen ImageNet image features may be computed once because "
                       "the encoder is not fitted to GRAPE images or labels."))
        blocks.append(("p", "For each endpoint, every training visit inherits its eye's released retrospective "
                       "progression label. This is weak supervision at the eye level; it does not provide a true "
                       "progression-state label for each visit. Inverse-visit-count sample weights give each training "
                       "eye equal total weight, with weights normalized to mean one. Each visit is scored independently "
                       "from its own image. Later held-out visits do not enter an earlier held-out visit's score."))
        blocks.append(("p", "The visit model uses a frozen ImageNet ResNet152 encoder with 2,048 output features and "
                       "an XGBoost binary logistic classifier. The paper-based defaults are 100 trees, maximum depth 6, "
                       "and learning rate 0.05; " + tree_description + " are implementation "
                       "choices. This image-only adaptation does not synthesize unavailable text or human assessments. "
                       "The archived model configuration records the exact settings used in this run."))
        blocks.append(("p", "Within each outer training partition, inner patient-grouped cross-fitting produces "
                       "visit scores for training eyes from XGBoost models that never saw those patients. A uses the "
                       "last score with a logistic progression mapping. B uses the last score, last-minus-first score, "
                       "ordinary least-squares score slope per actual elapsed year, mean score, and the fraction of "
                       "visits with scores strictly above the prespecified persistence threshold. Scaling and both "
                       "logistic mappings are fitted only to these inner out-of-fold summaries. A separate XGBoost "
                       "model is then fitted to all outer training patients and applied to the outer held-out patients; "
                       "A and B share that model and its visit scores. Both comparators therefore use a visit model "
                       "trained on all eligible training visits; 'latest visit' refers to A's inputs at assessment time. "
                       "Outer held-out labels never enter model fitting; labels are used for class-stratified partitioning "
                       "and final evaluation. "
                       "No hyperparameters are selected from outer test performance, and no future prediction horizon "
                       "is claimed."))
    else:
        blocks.append(("p", "Each eye is a longitudinal unit. All visits and both eyes from a patient remain in the same held-out fold. The fixed diagnosis detector is applied separately at each visit; later visits do not enter earlier visit scores. Outcome labels and visual-field measurements are excluded from the visit detector and temporal feature inputs. Original CFP files and contemporaneous IOP, when explicitly enabled in the fixed model, are the available GRAPE predictors; no baseline-only measurement is copied to later visits. Raw data are read without modification."))
        blocks.append(("p", "A uses the last score with a logistic progression mapping. B uses the last score, last-minus-first score, ordinary least-squares score slope per actual elapsed year, mean score, and the fraction of visits with scores strictly above the prespecified persistence threshold. Both use logistic regression with training-fold feature scaling; all preprocessing and fitting occur within training patients. The progression mappings are internally cross-validated on GRAPE even when the fixed diagnosis detector is external. No future prediction horizon is claimed."))
    solver_description = (
        "a float64 GPU damped-Newton solver with Armijo line search, at most 2,000 iterations and an absolute "
        "gradient infinity-norm tolerance of 0.0001. Training-only scaling reproduces StandardScaler's population "
        "variance and numerical-constant rule. The objective includes the intercept in the L2 penalty, matching "
        "binary liblinear with intercept_scaling=1; the solver and stopping rule differ, so bitwise-equivalent "
        "coefficients are not assumed. "
        if gpu_statistics else
        "the liblinear solver, at most 2,000 iterations, tolerance 0.0001, and training-only StandardScaler. ")
    blocks.append(("p", "Both progression mappings use L2-regularized logistic regression with balanced class weights, "
                   + solver_description +
                   "Classification uses an output of at least 0.5. Balanced-class outputs are not clinically calibrated "
                   "probabilities. " +
                   ("StratifiedKFold operates on unique patients, stratified by whether any included eye has the "
                    "endpoint, and then expands patient partitions back to their eyes. Requested outer and inner "
                    "fold counts are reduced only for class feasibility; an endpoint is not estimable if no valid "
                    "nested partition is available at the fixed seed. "
                    if internal else
                    "StratifiedGroupKFold shuffles patients using the recorded fixed seed; the requested fold count "
                    "is capped by positive- and negative-patient counts and reduced if necessary until every training "
                    "and test partition contains both classes. ") +
                   "No seed or fold choice is selected by predictive performance. A and B share the identical "
                   "splits within each endpoint."))
    if gpu_statistics:
        device_assignment = (
            "Independent endpoints are assigned to the recorded GPU devices, with one endpoint at a time per device. "
            if config.get("endpoint_compute_devices") else
            "Numerical devices are recorded with the fitted heads and bootstrap intervals. ")
        blocks.append(("p", "Temporal feature arithmetic, logistic fitting, metric calculations, paired patient "
                       "resampling and percentile confidence intervals use PyTorch CUDA float64. " + device_assignment +
                       "Bootstrap uses a seeded device-specific Torch generator, so its draws differ from the "
                       "previous NumPy PCG64 implementation even at the same seed. Metric definitions, paired "
                       "patient clustering and percentile interpolation are unchanged. CPU code handles input/output, "
                       "patient-split bookkeeping, task orchestration and report rendering. GPU tree building and "
                       "the new numerical solver are recorded implementation changes; numerical identity with a "
                       "CPU run is not claimed."))
    settings = ["n_splits", "seed", "bootstrap_replicates", "persistence_threshold", "logistic_c", "decision_threshold"]
    if internal:
        settings.insert(1, "inner_splits")
    if gpu_statistics:
        settings.extend(["logistic_solver", "gpu_device_ids", "endpoint_compute_devices"])
    blocks.append(_table(["Analysis setting", "Value"], [[key, str(config.get(key, "not recorded"))] for key in settings]))
    fold_rows = []
    for ep in ENDPOINTS:
        result = evaluation.get("endpoints", {}).get(ep, {})
        bs = result.get("bootstrap", {})
        fold_rows.append([ENDPOINT_NAMES[ep], str(result.get("status", "not_estimable")), _count(result.get("n_splits")),
                          _count(result.get("n_positive_patients")), _count(bs.get("valid")), _count(bs.get("requested")),
                          _count(bs.get("skipped_single_class"))])
    blocks.append(_table(["Endpoint", "Status", "Actual folds", "Positive patients", "Valid bootstrap draws", "Requested draws", "Single-class draws skipped"], fold_rows))
    if internal:
        blocks.append(("p", "Nested partitions contain fewer independent progression-positive patients than the full "
                       "cohort. An inner training partition may contain only one positive patient even when both "
                       "classes are technically present; such fits can be highly unstable. Per-fold patient identities, "
                       "class counts, and actual inner splits are archived in evaluation.json. A feasible split does "
                       "not establish adequate statistical precision."))
    blocks.append(("p", "Bootstrap draws resample patients, retain their eyes together, and use paired A/B predictions. Single-class draws cannot estimate all classification metrics and are skipped. The percentile intervals condition on already fitted out-of-fold predictions: the models and folds are not refitted in each draw, so training-pipeline uncertainty is not covered. AUPRC is implemented as average precision. Endpoint-specific folds, training counts, and any available regression coefficients are preserved in evaluation.json. Regression coefficients describe fitted associations, not clinical causation."))
    blocks.append(("h2", "Model provenance and limits"))
    blocks.append(("p", "The current software is a paper-based GlaBoost reconstruction adapted to available GRAPE modalities. It must not be described as the authors' released implementation, an independently validated clinical tool, or a reproduction of the paper's reported accuracy. " +
                   ("The original diagnostic task has been changed to three separate progression endpoints. This "
                    "report evaluates the complete fitted pipeline internally on GRAPE; it is not an external "
                    "validation of a pretrained GlaBoost diagnosis model. The eye-level labels provide weak supervision "
                    "over the original follow-up window and do not establish visit-level disease state or progression timing."
                    if internal else
                    "GRAPE contains glaucoma eyes only, so its progression labels cannot train or validate a normal-versus-glaucoma diagnosis classifier.")))
    blocks.append(("code", json.dumps(_compact_provenance(provenance), indent=2, ensure_ascii=False, allow_nan=False)))
    blocks.append(("p", ("The pretrained encoder's provenance is recorded separately from the models fitted on GRAPE. "
                       "Inner cross-fitted training scores and scores from the model fitted on all outer training "
                       "patients can have different distributions; this transfer is part of the evaluated pipeline. "
                       if internal else
                       "Any independence statement above is based on supplied provenance, not an independent audit. ") +
                   "Complete-case image eligibility and sparse progression-positive patients can limit generalizability and precision. No benefit is assumed in advance, no p-values are used, and these results do not establish prospective clinical utility. This simple temporal aggregation does not establish persistent, traceable longitudinal clinical reasoning."))
    blocks.append(("h2", "Eligibility exclusions"))
    exclusions = cohort.get("exclusions", [])
    reason_counts = Counter(row.get("reason", "Unspecified") for row in exclusions)
    blocks.append(_table(["Exclusion reason", "Eyes"], [[reason, count] for reason, count in sorted(reason_counts.items())]))
    if not exclusions:
        blocks.append(("p", "No eye-level exclusions were recorded in this run."))
    else:
        blocks.append(("p", "The complete per-eye exclusion list, with available CFP counts, is retained in exclusions.csv and evaluation.json."))
    blocks.append(("h2", "Files for checking and reuse"))
    blocks.append(("p", "report.html embeds both figures and can be shared as a standalone file. report.md, primary_results.csv, supplementary_metrics.csv, predictions.csv, temporal_features.csv, exclusions.csv, evaluation.json, and provenance.json retain the numeric results and analysis provenance. The study command marks a run complete in status.json only after all outputs are written. Figure files are available as PNG (300 dpi), PDF, and SVG. Per-eye outputs are research audit material, not individual patient reports."))
    return blocks


def _markdown(blocks):
    lines = []
    for block in blocks:
        kind = block[0]
        if kind in ("h1", "h2"):
            lines.append(("# " if kind == "h1" else "## ") + block[1])
        elif kind == "p":
            lines.append(block[1])
        elif kind == "code":
            lines.append("```json\n" + block[1] + "\n```")
        elif kind == "image":
            lines.append("![" + block[2] + "](" + block[1] + ")\n\n" + block[2])
        elif kind == "table":
            def row(values):
                return "| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values) + " |"
            lines.append("\n".join([row(block[1]), row(["---"] * len(block[1]))] + [row(v) for v in block[2]]))
    return "\n\n".join(lines) + "\n"


def _html(blocks, run_dir):
    body = []
    for block in blocks:
        kind = block[0]
        if kind in ("h1", "h2", "p"):
            body.append("<{0}>{1}</{0}>".format(kind, html.escape(block[1])))
        elif kind == "code":
            body.append("<pre>" + html.escape(block[1]) + "</pre>")
        elif kind == "image":
            image = base64.b64encode((run_dir / block[1]).read_bytes()).decode("ascii")
            body.append('<figure><img alt="{}" src="data:image/png;base64,{}"><figcaption>{}</figcaption></figure>'.format(html.escape(block[2], quote=True), image, html.escape(block[2])))
        elif kind == "table":
            head = "".join("<th>" + html.escape(str(v)) + "</th>" for v in block[1])
            rows = "".join("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in row) + "</tr>" for row in block[2])
            body.append("<div class=table-wrap><table><thead><tr>" + head + "</tr></thead><tbody>" + rows + "</tbody></table></div>")
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>' + html.escape(blocks[0][1]) + '</title><style>'
            'body{max-width:1080px;margin:40px auto;padding:0 24px;color:#20262c;font:16px/1.55 system-ui,sans-serif}'
            'h1{font-size:28px;line-height:1.25}h2{font-size:21px;margin-top:2em}'
            'table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd3d9}'
            'th{background:#eef2f5}.table-wrap{overflow-x:auto}figure{margin:24px 0}img{width:100%;height:auto}'
            'figcaption{font-size:13px;color:#46515c}pre{padding:16px;background:#eef2f5;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}'
            '@media print{body{margin:0}figure,table{break-inside:avoid}h2{break-after:avoid}}'
            '</style></head><body>' + "\n".join(body) + '</body></html>\n')


def write_report(run_dir, evaluation, cohort, provenance, eye_records, synthetic=False):
    """Write a complete cohort report into an existing, new run directory.

    ``eye_records`` contains dictionaries with eye_id, patient_id, times (years),
    and scores. For internal nested validation, these are outer held-out records
    with endpoint and fold (or outer_fold); an eye may occur once per endpoint.
    The study caller owns the status.json completion marker and
    additional archived inputs, including exclusions.csv.
    Existing report artifacts are never overwritten.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError("run_dir must be an existing new run directory")
    parts = run_dir.resolve().parts
    if any(parts[i:i + 2] == ("data", "raw") for i in range(len(parts) - 1)):
        raise ValueError("Reports must not be written inside data/raw")
    names = ["report.md", "report.html", "primary_results.csv", "supplementary_metrics.csv",
             "predictions.csv", "temporal_features.csv", "evaluation.json", "provenance.json"]
    names.extend("figures/" + stem + "." + ext for stem in ("primary_comparison", "score_trajectories") for ext in ("png", "pdf", "svg"))
    if any((run_dir / name).exists() for name in names):
        raise FileExistsError("Report artifacts already exist; choose a new run directory")
    evaluation, cohort, provenance, eye_records = map(_clean, (evaluation, cohort, provenance, eye_records))
    provenance = dict(provenance, synthetic=bool(synthetic))
    evaluation = dict(evaluation, synthetic=bool(synthetic))
    # Normalize all content before touching disk, so nonfinite values cannot leak
    # into plots, CSVs, or the portable report.
    json.dumps([evaluation, cohort, provenance, eye_records], allow_nan=False)
    _save_figures(run_dir, evaluation, eye_records, synthetic,
                  internal=provenance.get("validation_design") == "internal_nested_patient_cv")
    blocks = _report_blocks(evaluation, cohort, provenance, eye_records, synthetic)
    primary, supplementary = _primary_rows(evaluation), _supplementary_rows(evaluation)
    predictions, features = evaluation.get("predictions", []), evaluation.get("features", [])
    for filename, rows, defaults in (
        ("primary_results.csv", primary, ("endpoint", "status")),
        ("supplementary_metrics.csv", supplementary, ("endpoint", "method", "metric")),
        ("predictions.csv", predictions, ("endpoint", "eye_id", "patient_id", "fold", "y_true", "latest_probability", "longitudinal_probability")),
        ("temporal_features.csv", features, ("eye_id", "patient_id")),
    ):
        _write_csv(run_dir / filename, [dict(row, synthetic=bool(synthetic)) for row in rows], tuple(defaults) + ("synthetic",))
    for filename, value in (("evaluation.json", dict(evaluation, cohort=cohort)), ("provenance.json", provenance)):
        with (run_dir / filename).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    for filename, content in (("report.md", _markdown(blocks)), ("report.html", _html(blocks, run_dir))):
        with (run_dir / filename).open("x", encoding="utf-8") as stream:
            stream.write(content)
    return run_dir / "report.html"
