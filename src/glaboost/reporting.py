"""Cohort-level, auditable reports from completed longitudinal evaluations.

No fitting, clinical interpretation, or result fabrication happens here. HTML
embeds its figures so the report can be shared without its companion directory.
"""

import base64
from collections import Counter
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ENDPOINTS = ("plr2", "plr3", "md_slope")
ENDPOINT_NAMES = {"plr2": "PLR2", "plr3": "PLR3", "md_slope": "MD slope"}
METRICS = ("balanced_accuracy", "auroc", "auprc", "sensitivity", "specificity", "f1")
METHOD_NAMES = {"latest": "Latest visit", "longitudinal": "Longitudinal integration"}
BLUE, ORANGE = "#0072B2", "#E69F00"


def embed_report_links(report_path):
    """Bundle this project's generated HTML/CSV/JSON links into one offline page.

    Child HTML already embeds its figures. Only the overview's presentation is
    rewritten; linked files, numeric records and archived code stay unchanged.
    Repeated calls can add a source report without duplicating existing sections.
    """
    report_path = Path(report_path).resolve()
    page = report_path.read_text(encoding="utf-8")
    anchor = re.compile(r'(<a\b[^>]*\bhref=)([\"\x27])([^\"\x27]+)\2', re.IGNORECASE)
    body = re.compile(r'<body\b[^>]*>(.*?)</body>', re.IGNORECASE | re.DOTALL)
    if not body.search(page):
        raise ValueError(f"Generated report has no HTML body: {report_path}")
    existing_ids = set(re.findall(r'\bid=[\"\x27]([^\"\x27]+)[\"\x27]', page))
    visited = {report_path: "report-top"}
    sections = []

    def rewrite(content, directory):
        def replace(match):
            href = html.unescape(match[3])
            url = urlsplit(href)
            if url.scheme or url.netloc or not url.path:
                return match[0]
            target = (directory / unquote(url.path)).resolve()
            if target.suffix.lower() not in (".html", ".csv", ".json"):
                return match[0]
            relative = os.path.relpath(target, report_path.parent)
            section_id = visited.get(target)
            if section_id is None:
                section_id = "attachment-" + hashlib.sha256(relative.encode()).hexdigest()[:16]
                visited[target] = section_id  # Register before following any backlinks.
                if section_id not in existing_ids:
                    raw = target.read_text(encoding="utf-8")
                    if target.suffix.lower() == ".html":
                        child = body.search(raw)
                        if child is None:
                            raise ValueError(f"Generated report has no HTML body: {target}")
                        embedded = rewrite(child[1], target.parent)
                    elif target.suffix.lower() == ".csv":
                        with target.open(newline="", encoding="utf-8") as stream:
                            rows = list(csv.reader(stream))
                        rendered = []
                        for index, row in enumerate(rows):
                            tag = "th" if index == 0 else "td"
                            rendered.append("<tr>" + "".join(
                                f"<{tag}>{html.escape(cell)}</{tag}>" for cell in row) + "</tr>")
                        embedded = '<div class="attachment-data"><table>' + "".join(rendered) + "</table></div>"
                    else:
                        embedded = '<pre class="attachment-data">' + html.escape(raw) + "</pre>"
                    sections.append(
                        f'<section class="report-attachment" id="{section_id}">'
                        f'<h2>{html.escape(relative)}</h2><p><a href="#report-top">Back to overview</a></p>'
                        + embedded + "</section>")
            return match[1] + match[2] + "#" + section_id + match[2]
        return anchor.sub(replace, content)

    page = rewrite(page, report_path.parent)
    if "report-top" not in existing_ids:
        page = re.sub(r'(<body\b[^>]*>)', r'\1<div id="report-top"></div>', page, count=1)
    if "portable-report-style" not in existing_ids:
        page = page.replace("</body>", '<style id="portable-report-style">'
            '.report-attachment{margin-top:48px;padding-top:20px;border-top:2px solid #ccd5df}'
            '.report-attachment h2{overflow-wrap:anywhere}.report-attachment img{max-width:100%;height:auto}'
            '.report-attachment figure{margin:24px 0}.report-attachment .table-wrap{overflow-x:auto}'
            '.report-attachment pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#eef2f5}'
            '.attachment-data{max-height:32rem;overflow:auto}'
            '@media print{.attachment-data{max-height:none;overflow:visible}}'
            '</style></body>', 1)
    if "portable-report-note" not in existing_ids:
        page = re.sub(r'(</h1>)', r'\1<p id="portable-report-note">'
            'This single HTML includes the linked reports, figures, numeric tables and audit records. '
            'Links below jump to sections within this page; no companion files are needed to view them.</p>',
            page, count=1)
    page = page.replace("</body>", "\n".join(sections) + "</body>", 1)
    report_path.write_text(page, encoding="utf-8")


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


def _visit_model_description(evaluation, provenance):
    """Describe the fixed detector's saved settings without inventing provenance."""
    info = provenance.get("model_info", {}) or {}
    config = info.get("config", {}) or {}
    spec = info.get("encoder_spec", {}) or (info.get("encoder_specs") or {}).get("image", {}) or {}
    encoder = config.get("image_encoder", spec.get("encoder", "not recorded"))
    encoder_name = "ResNet152" if encoder == "resnet152" else str(encoder)
    dimension = spec.get("output_dim", "not recorded")
    dimension = format(dimension, ",") if isinstance(dimension, int) else str(dimension)
    keys = ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha")
    settings = {key: config.get(key, "not recorded") for key in keys}
    # These two constants are part of this identified implementation, rather
    # than defaults inferred for an arbitrary imported detector.
    if info.get("implementation") == "GlaBoost paper-based reconstruction; not author weights":
        settings["reg_lambda"] = config.get("reg_lambda", 1.)
        settings["reg_alpha"] = config.get("reg_alpha", 0.)
    source = spec.get("source", {}) or {}
    official = ("ResNet152_Weights.IMAGENET1K_V1", "394f9c45") if encoder == "resnet152" else None
    imagenet = bool(official and source.get("kind") == "torchvision"
                    and source.get("weights") == official[0]
                    and str(source.get("sha256", "")).startswith(official[1]))
    qualifier = "frozen " if spec.get("frozen") is True else "fixed "
    if imagenet:
        qualifier += "ImageNet "
    text = (
        "The visit model uses a {}{} encoder with {} output features and an XGBoost binary "
        "logistic classifier: {} trees, maximum depth {}, learning rate {}, row subsampling {}, column "
        "subsampling {}, L2 {}, and L1 {}."
    ).format(qualifier, encoder_name, dimension, settings["n_estimators"], settings["max_depth"],
             settings["learning_rate"], settings["subsample"], settings["colsample_bytree"],
             settings["reg_lambda"], settings["reg_alpha"])
    preprocessing = spec.get("preprocessing", {}) or {}
    if preprocessing:
        mean, std = preprocessing.get("mean"), preprocessing.get("std")
        normalization = ("normalization not recorded" if "mean" not in preprocessing or "std" not in preprocessing else
                         "no mean/std normalization" if mean is None and std is None else
                         "ImageNet mean/std normalization" if mean == [.485, .456, .406] and std == [.229, .224, .225] else
                         "normalization with mean {} and std {}".format(mean, std))
        text += " Image preprocessing: color {}; resize {}; interpolation {}; scale {}; {}.".format(
            preprocessing.get("color", "not recorded"), preprocessing.get("resize", "not recorded"),
            preprocessing.get("interpolation", "not recorded"), preprocessing.get("scale", "not recorded"), normalization)
    else:
        text += " Image preprocessing is not recorded in the supplied metadata."
    if not imagenet:
        text += " Encoder training origin is not established by the checkpoint hash alone; see the supplied provenance."
    return text


def _external_design(provenance):
    design = provenance.get("validation_design")
    if design not in ("external_fixed_detector", "retrospective_fixed_detector_unverified_external"):
        raise ValueError("Reports require fixed-detector validation provenance.")
    return design == "external_fixed_detector"


def _summary(evaluation, cohort, provenance, synthetic):
    external = _external_design(provenance)
    opening = (
        "We conducted a retrospective external evaluation of a fixed visit-level detector on the GRAPE cohort, "
        "with internally cross-validated progression mappings. External independence is based on the supplied "
        "detector provenance declaration. "
        if external else
        "We conducted a retrospective evaluation on the GRAPE longitudinal glaucoma cohort. "
        "External validation is not established because independence of the visit-level detector from GRAPE "
        "has not been documented. "
    )
    opening += _visit_model_description(evaluation, provenance) + " "
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


def cohort_template_summary(evaluation, cohort, provenance, synthetic=False):
    """Fill the professor's narrative from saved results without assuming benefit."""
    external = _external_design(provenance)
    text = "SYNTHETIC TEST DATA — SOFTWARE CHECK ONLY. " if synthetic else ""
    text += (
        "We conducted a focused retrospective evaluation on the publicly available GRAPE longitudinal "
        "glaucoma cohort using a fixed image-only GlaBoost-style detector. "
    )
    text += ("Detector training was independent of GRAPE according to the recorded provenance. " if external else
             "Independence of detector training from GRAPE has not been established. ")
    text += (
        "The progression mappings were trained and patient-cross-validated within GRAPE. "
        "Of {} source eyes, {} eyes from {} patients met prespecified eligibility criteria, providing {} "
        "evaluable visits, a median of {} visits per eye and a median follow-up of {} months between the first "
        "and last eligible fundus photographs. Progression was evaluated separately using GRAPE's three "
        "prespecified visual-field criteria (PLR2, PLR3, and MD slope). The fixed detector was applied "
        "independently at each eligible visit, and its outputs were integrated using prespecified latest "
        "score, change, slope, mean, and persistence summaries, without a persistent longitudinal "
        "representation or reasoning model. "
    ).format(_count(cohort.get("source", {}).get("n_eyes")), _count(cohort.get("n_eyes")),
             _count(cohort.get("n_patients")), _count(cohort.get("n_visits")),
             _number(cohort.get("visits_per_eye", {}).get("median"), 1),
             _number(cohort.get("followup_months", {}).get("median"), 1))
    deltas = []
    for endpoint in ENDPOINTS:
        result = evaluation.get("endpoints", {}).get(endpoint, {})
        if result.get("status") != "ok":
            text += f"{ENDPOINT_NAMES[endpoint]} was not estimable: {result.get('reason', 'no valid evaluation')}. "
            continue
        metrics = result["metrics"]
        delta = result["delta_balanced_accuracy"]
        percent = lambda value, signed=False: _number(100 * value if value is not None else None, 1, signed)
        difference = percent(delta.get("estimate"), True)
        interval = (f"{percent(delta['ci_low'], True)} to {percent(delta['ci_high'], True)}"
                    if delta.get("ci_low") is not None and delta.get("ci_high") is not None else "not estimable")
        text += (
            f"For {ENDPOINT_NAMES[endpoint]}, balanced accuracy changed from "
            f"{percent(metrics['latest']['balanced_accuracy'].get('estimate'))}% with the latest visit alone to "
            f"{percent(metrics['longitudinal']['balanced_accuracy'].get('estimate'))}% with longitudinal integration "
            f"(Δ = {difference} percentage points, 95% CI {interval}). "
        )
        deltas.append(delta)
    if len(deltas) == 3 and all(d.get("estimate") is not None and d["estimate"] > 0 for d in deltas):
        text += "Point estimates favored longitudinal integration across all three progression definitions. "
    if len(deltas) == 3 and all(d.get("ci_low") is not None and d.get("ci_high") is not None
                                and d["ci_low"] <= 0 <= d["ci_high"] for d in deltas):
        text += "However, all three paired 95% confidence intervals included zero; a consistent benefit has not been established. "
    elif len(deltas) == 3 and all(d.get("ci_low") is not None and d["ci_low"] > 0 for d in deltas):
        text += "The paired intervals were above zero for all three endpoints, supporting higher balanced accuracy within this analysis. "
    else:
        text += "The results do not establish a consistent gain across all three progression definitions. "
    return text + (
        "Intervals used paired patient-cluster bootstrap conditional on fixed out-of-fold predictions, "
        "without refitting uncertainty or multiple-comparison adjustment. These results concern retrospective "
        "progression assessment, not future prediction or independent external validation of the complete "
        "progression pipeline. The central UG3 challenge remains unresolved: transforming episodic AI evidence "
        "into persistent, traceable, uncertainty-aware longitudinal clinical intelligence."
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


def _selected_eyes(eye_records):
    ordered = sorted(eye_records, key=lambda row: (str(row.get("patient_id", "")), str(row.get("eye_id", ""))))
    return ordered[:6]


def _trajectory_figure(eye_records, synthetic=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    selected = _selected_eyes(eye_records)
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
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, max(1, max_time) * 1.03)
            ax.set_yticks([0, .5, 1])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
            ax.grid(color="#E5E5E5", linewidth=.6)
            if index // 3 == 1:
                ax.set_xlabel("Time from baseline (years)", fontsize=8)
            if index % 3 == 0:
                ax.set_ylabel("Visit-level diagnosis score")
        if not selected:
            fig.text(.5, .5, "No eligible score trajectories", ha="center", va="center")
        fig.suptitle("SYNTHETIC TEST DATA — NOT STUDY RESULTS" if synthetic else "Illustrative visit-level score histories", y=.98, fontsize=11)
        fig.text(.02, .15, "First six eligible eyes sorted by patient and eye ID; selection does not use outcomes or performance.", fontsize=8)
        fig.text(.02, .060, "Points are observed visit scores; lines guide the eye. Score changes alone do not establish progression.", fontsize=8)
        return fig


def _save_figures(run_dir, evaluation, eye_records, synthetic):
    import matplotlib.pyplot as plt

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    for name, builder, values in (("primary_comparison", _primary_figure, evaluation),
                                   ("score_trajectories", _trajectory_figure, eye_records)):
        fig = builder(values, synthetic=synthetic)
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
    external = _external_design(provenance)
    title = (
        "Preliminary external evaluation of longitudinal glaucoma progression assessment" if external else
        "Preliminary retrospective assessment of longitudinal glaucoma progression")
    if synthetic:
        title = "SYNTHETIC TEST REPORT — " + title
    blocks = [("h1", title)]
    if synthetic:
        blocks.append(("p", "SYNTHETIC TEST DATA. This document checks the software and must not be presented as research findings."))
    blocks.append(("p", "Run: {}. Created: {}.".format(provenance.get("run_name", "not recorded"), provenance.get("created_at", "not recorded"))))
    if not external:
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
         "Fixed detector input; original photographs only, no annotation overlays"],
        ["Contemporaneous IOP", "{} available; {} missing in included visits".format(
            _count(iop.get("available_visits")), _count(iop.get("missing_visits"))),
         "Enabled in fixed detector" if iop.get("selected") else "Not enabled in fixed detector"],
        ["Clinical text / human risk assessment", "Unavailable as longitudinal GRAPE inputs", "Not used; not synthesized"],
        ["Baseline OCT / RNFL", "Baseline-only measurements", "Not copied across subsequent visits; not used"],
        ["Visual field / PLR2, PLR3, MD-slope labels", "Released progression reference labels",
         "Outcomes only; excluded from predictors"],
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
                   "Illustrative histories selected by sorted patient and eye ID, without outcome or performance selection. "
                   "No diagnostic interpretation of fundus anatomy is inferred from these plots."))
    selected = _selected_eyes(eye_records)
    trajectory_rows = []
    for eye in selected:
        scores, times = eye.get("scores", []), eye.get("times", [])
        trajectory_rows.append([str(eye.get("eye_id", "")), str(len(scores)),
                                _number(max(times) - min(times), 2) if times else "not recorded",
                                _number(max(scores) - min(scores)) if scores else "not recorded"])
    blocks.append(_table(["Illustrative eye", "Visits", "Observed span (years)", "Score range (maximum − minimum)"], trajectory_rows))
    blocks.append(("p",
                   "A nearly flat trajectory indicates little change in the detector's score, which may reflect stable appearance or limited sensitivity to progression. A changing trajectory can also reflect image acquisition variability. Neither pattern establishes clinical stability or deterioration. Diagnosis scores are not validated severity measurements or future progression risks. These examples illustrate inputs to the temporal summary, and do not replace cohort-level reference-label evaluation."))
    blocks.append(("h2", "Prespecified analysis and reproducibility"))
    config = evaluation.get("config", {})
    gpu_statistics = config.get("logistic_solver") == "torch_newton"
    model_config = (provenance.get("model_info", {}) or {}).get("config", {}) or {}
    blocks.append(("p", _visit_model_description(evaluation, provenance)))
    blocks.append(("p", "The fixed detector's recorded tree method is {}; its training seed is {}. "
                   "It is not refitted using GRAPE progression labels. The same detector and visit scores "
                   "serve both A and B and all three reference outcomes.".format(
                       model_config.get("tree_method", "not recorded"),
                       model_config.get("random_state", "not recorded"))))
    blocks.append(("p", "Eligibility requires at least {} visits per eye with an available original CFP and a valid "
                   "visit score, ordered by actual elapsed years. Visits without a CFP are omitted; source cohort "
                   "visit counts are not treated as available image counts.".format(
                       _count(provenance.get("minimum_cfp_visits_per_eye")))))
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
                   "probabilities. "
                   "StratifiedGroupKFold shuffles patients using the recorded fixed seed; the requested fold count "
                   "is capped by positive- and negative-patient counts and reduced if necessary until every training "
                   "and test partition contains both classes. "
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
                       "patient-split bookkeeping, task orchestration and report rendering. Numerical identity "
                       "with a CPU run is not claimed."))
    settings = ["n_splits", "seed", "bootstrap_replicates", "persistence_threshold", "logistic_c", "decision_threshold"]
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
    blocks.append(("p", "Bootstrap draws resample patients, retain their eyes together, and use paired A/B predictions. Single-class draws cannot estimate all classification metrics and are skipped. The percentile intervals condition on already fitted out-of-fold predictions: the models and folds are not refitted in each draw, so training-pipeline uncertainty is not covered. AUPRC is implemented as average precision. Endpoint-specific folds, training counts, and any available regression coefficients are preserved in evaluation.json. Regression coefficients describe fitted associations, not clinical causation."))
    blocks.append(("h2", "Model provenance and limits"))
    blocks.append(("p", "The current software applies a fixed GlaBoost-compatible detector using modalities "
                   "available at individual GRAPE visits. Missing text, structured fundus descriptors, and human "
                   "assessments are not synthesized. This does not establish reproduction of the complete original "
                   "multimodal model or its reported accuracy. GRAPE contains glaucoma eyes only, so its progression "
                   "labels cannot train or validate a normal-versus-glaucoma diagnosis classifier. The temporal "
                   "progression mappings are fitted and patient-cross-validated on GRAPE; the complete progression "
                   "pipeline has not undergone independent external validation."))
    blocks.append(("code", json.dumps(_compact_provenance(provenance), indent=2, ensure_ascii=False, allow_nan=False)))
    blocks.append(("p", "Any independence statement above is based on supplied provenance, not an independent audit. "
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
    and scores from the same fixed detector; every eligible eye occurs once.
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
    _external_design(provenance)
    provenance = dict(provenance, synthetic=bool(synthetic))
    evaluation = dict(evaluation, synthetic=bool(synthetic))
    # Normalize all content before touching disk, so nonfinite values cannot leak
    # into plots, CSVs, or the portable report.
    json.dumps([evaluation, cohort, provenance, eye_records], allow_nan=False)
    _save_figures(run_dir, evaluation, eye_records, synthetic)
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
