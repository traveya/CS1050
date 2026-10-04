"""Compare CSVs beside this script; run: python compare_anonymization.py.

Install: python -m pip install pandas numpy matplotlib
Outputs: six PNG charts, six aggregate CSVs, and summary.md in outputs/.
All analyses describe ROWS, not independent people. No record-level linkage
between files is attempted. Missing forum counts are never interpreted as zero.
"""
from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLDER = Path(__file__).resolve().parent
ORIGINAL_FILE = "raw_data.csv"
ANONYMIZED_FILE = "anonymized_data.csv"
OUTPUT_FOLDER = "outputs"
MAX_CATEGORIES = 18
CHANGE_THRESHOLD_PP = 2.0  # Descriptive threshold, not a significance test.
MISSING = "[Missing]"
INVALID = "[Invalid / unrecognized]"
OTHER = "[Other categories]"
MISSING_TOKENS = {"", "null", "nan", "<na>"}


def clean(series):
    """Preserve 'NA' (Namibia), leading zeros, and education code 'none'."""
    s = series.fillna("").astype(str).str.strip()
    return s.mask(s.str.lower().isin(MISSING_TOKENS), MISSING)


def read_csv(filename):
    path = FOLDER / filename
    if not path.exists():
        raise FileNotFoundError(f"Place {filename} beside this script: {FOLDER}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def year_interval(value):
    """Accept YYYY, 199*, 19**, and inclusive bands such as 1980–1999.

    These are birth years, not ages. No survey date or exact age is assumed.
    Other formats are flagged rather than guessed.
    """
    if value == MISSING:
        return None
    if re.fullmatch(r"\d{4}", value):
        return int(value), int(value)
    if re.fullmatch(r"\d{1,3}\*{1,3}", value) and len(value) == 4:
        return int(value.replace("*", "0")), int(value.replace("*", "9"))
    match = re.fullmatch(r"(\d{4})\s*[-–—]\s*(\d{4})", value)
    if match and int(match[1]) <= int(match[2]):
        return int(match[1]), int(match[2])
    return None


def align_birth_years(frames):
    """Find common bins that never split any supplied generalized interval.

    Start with decades, then remove boundaries inside any source interval.
    Thus exact years align to 199* or 1980–1999 without inventing exact years.
    """
    intervals = []
    for frame in frames:
        if "YoB" in frame:
            intervals += [year_interval(v) for v in clean(frame.YoB).unique()]
    intervals = [v for v in intervals if v is not None]
    if not intervals:
        return lambda s: clean(s).map(lambda v: MISSING if v == MISSING else INVALID), "No parseable birth years."
    low = min(a for a, b in intervals) // 10 * 10
    high = (max(b for a, b in intervals) // 10 + 1) * 10
    boundaries = [x for x in range(low, high + 1, 10)
                  if not any(a < x <= b for a, b in intervals)]

    def convert(series):
        def label(v):
            if v == MISSING:
                return MISSING
            interval = year_interval(v)
            if interval is None:
                return INVALID
            i = np.searchsorted(boundaries, interval[0], side="right") - 1
            return f"{boundaries[i]}–{boundaries[i + 1] - 1}"
        return clean(series).map(label)

    generalized = any(a != b for a, b in intervals)
    note = ("Birth-year cohorts, not ages. Common bins never split an input band. "
            "Exact years are displayed at decade resolution unless broader input bands require further merging. "
            "Missing and unrecognized values remain separate.")
    if generalized:
        note += " Generalization prevents recovering within-band differences."
    return convert, note


EDUCATION = {
    "p": "Doctorate", "p_se": "Doctorate", "p_oth": "Doctorate",
    "m": "Master’s / professional", "b": "Bachelor’s", "a": "Associate",
    "hs": "High school", "jhs": "Junior high", "el": "Elementary",
    "none": "No formal education", "other": "Other education",
}
BROAD_EDUCATION = {
    "Doctorate": "Postgraduate", "Master’s / professional": "Postgraduate",
    "Bachelor’s": "Undergraduate", "Associate": "Undergraduate",
    "High school": "School or less", "Junior high": "School or less",
    "Elementary": "School or less", "No formal education": "School or less",
}
# Edit these aliases if your anonymizer uses different labels.
EDUCATION_ALIASES = {
    "postgraduate": "Postgraduate", "postgraduate degree": "Postgraduate",
    "undergraduate": "Undergraduate", "undergraduate degree": "Undergraduate",
    "associate_or_bachelor": "Undergraduate", "school_or_less": "School or less",
    "school or less": "School or less",
}


def align_education(frames):
    def canonical(series):
        return clean(series).map(lambda v: EDUCATION.get(v, EDUCATION_ALIASES.get(v.lower(), v)))
    broad = any("LoE" in f and canonical(f.LoE).isin(set(BROAD_EDUCATION.values())).any()
                for f in frames)
    def convert(series):
        s = canonical(series)
        return s.replace(BROAD_EDUCATION) if broad else s
    note = "Legacy doctorate codes are combined in both files. Other education and missing education stay separate."
    if broad:
        note += " Both files use school-or-less, undergraduate, and postgraduate groups; degree-level distinctions are lost."
    return convert, note


def missing_reason(frame, columns):
    missing = [c for c in columns if c not in frame]
    return f"Analysis unavailable: {', '.join(missing)} removed" if missing else None


def numeric_posts(series):
    n = pd.to_numeric(clean(series).replace(MISSING, np.nan), errors="coerce")
    # A post count must be finite, nonnegative, and an integer.
    return n.where(np.isfinite(n) & n.ge(0) & n.mod(1).eq(0))


def common_categories(series_pair, ordered=False):
    available = [s for s in series_pair if s is not None]
    if not available:
        return [], series_pair, False
    counts = pd.concat(available, ignore_index=True).value_counts()
    special = [v for v in [MISSING, INVALID] if v in counts.index]
    regular = [v for v in counts.index if v not in special]
    selected = regular[:max(1, MAX_CATEGORIES - len(special))]
    if ordered:
        selected.sort()
    collapsed = len(selected) < len(regular)
    categories = selected + ([OTHER] if collapsed else []) + special
    permitted = set(selected + special)
    result = [None if s is None else s.where(s.isin(permitted), OTHER) for s in series_pair]
    return categories, result, collapsed


def aggregate(frame, groups, categories, participation):
    counts = groups.value_counts().reindex(categories, fill_value=0)
    result = pd.DataFrame({"category": categories, "rows": counts.values})
    result["percent_of_rows"] = counts.values / len(frame) * 100 if len(frame) else np.nan
    if participation:
        work = pd.DataFrame({"group": groups, "posts": numeric_posts(frame.nforum_posts)})
        by = work.groupby("group", dropna=False).posts
        result["valid_post_rows"] = by.count().reindex(categories, fill_value=0).values
        result["missing_or_invalid_post_rows"] = result.rows - result.valid_post_rows
        result["mean_posts"] = by.mean().reindex(categories).values
        rates = work.assign(active=work.posts.gt(0).where(work.posts.notna())).groupby("group").active.mean()
        result["percent_posting_among_valid"] = rates.reindex(categories).values * 100
    return result


def draw_and_summarize(spec, frames, labels, out):
    slug, title, column, convert, note, participation = spec
    required = [column] + (["nforum_posts"] if participation else [])
    reasons = [missing_reason(f, required) for f in frames]
    # Unknown new category schemes are not silently treated as equivalent.
    if not any(reasons) and column in ("LoE", "gender", "cc_by_ip"):
        before, after = [convert(f[column]) for f in frames]
        novel = set(after) - set(before) - {MISSING}
        if novel:
            reasons[1] = ("Analysis unavailable: unrecognized category changes in " + column +
                          ". Add an explicit common mapping; values: " + ", ".join(sorted(novel)[:8]))
    raw = [None if reason else convert(f[column]) for f, reason in zip(frames, reasons)]
    categories, grouped, collapsed = common_categories(raw, ordered=column == "YoB")
    if collapsed:
        note += " Less frequent categories are pooled identically in both panels, selected from combined row counts."
    tables = [None if g is None else aggregate(f, g, categories, participation)
              for f, g in zip(frames, grouped)]
    metric = "mean_posts" if participation else "percent_of_rows"
    xlabel = "Mean forum posts per row with a valid count" if participation else "Percentage of all rows"
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(categories) * .39 + 2)), sharex=True, sharey=True)
    maxima = [t[metric].max() for t in tables if t is not None and t[metric].notna().any()]
    limit = max(maxima, default=1)
    limit = max(limit, .1) * 1.4
    for ax, frame, label, reason, table in zip(axes, frames, labels, reasons, tables):
        ax.set_title(f"{label} · {len(frame):,} rows")
        ax.set_xlim(0, limit)
        ax.set_xlabel(xlabel)
        ax.spines[["top", "right"]].set_visible(False)
        if reason:
            ax.text(.5, .5, textwrap.fill(reason, 48), transform=ax.transAxes, ha="center", va="center")
            continue
        values = table[metric].to_numpy(dtype=float)
        y = np.arange(len(categories))
        ax.barh(y, np.nan_to_num(values), color="#3676a8" if label == "Original" else "#d08038")
        ax.set_yticks(y, categories)
        ax.tick_params(axis="y", labelleft=True)
        for i, row in table.iterrows():
            value = row[metric]
            if participation:
                label_text = f"{value:.2f}; n={int(row.valid_post_rows):,}" if pd.notna(value) else "No valid counts"
            else:
                label_text = f"{value:.1f}%; n={int(row.rows):,}" if pd.notna(value) else "No rows"
            ax.text((value if pd.notna(value) else 0) + limit * .012, i, label_text, va="center", fontsize=8)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=15)
    fig.text(.02, .01, textwrap.fill(note, 160), fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    fig.savefig(out / f"{slug}.png", dpi=160)
    plt.close(fig)
    exports = [t.assign(dataset=label) for label, t in zip(labels, tables) if t is not None]
    if exports:
        pd.concat(exports, ignore_index=True).to_csv(out / f"{slug}.csv", index=False)
    report = [f"## {title}", note]
    for label, reason, table in zip(labels, reasons, tables):
        if reason:
            report.append(f"{label}: {reason}.")
        elif table[metric].notna().any():
            top = table.loc[table[metric].idxmax()]
            unit = "mean posts" if participation else "% of rows"
            report.append(f"{label}: highest displayed category is {top.category}: {top[metric]:.2f} {unit}.")
        else:
            report.append(f"{label}: no usable observations for this metric.")
    if all(t is not None for t in tables):
        a, b = tables
        valid = a[metric].notna() & b[metric].notna()
        if valid.any():
            delta = b.loc[valid, metric] - a.loc[valid, metric]
            index = delta.abs().idxmax()
            unit = "posts per row" if participation else "percentage points"
            report.append(f"Largest absolute change among jointly observable categories: {a.loc[index, 'category']}, {delta.loc[index]:+.2f} {unit}.")
            if participation:
                report.append("Insight remains evaluable at this granularity; inspect mean changes alongside valid sample sizes. No statistical significance is claimed.")
            else:
                status = "changes" if delta.abs().max() >= CHANGE_THRESHOLD_PP else "is broadly preserved"
                report.append(f"Displayed composition {status} under a descriptive {CHANGE_THRESHOLD_PP:g}-percentage-point threshold.")
        else:
            report.append("Comparison cannot be evaluated: no jointly observable values.")
    else:
        report.append("The before/after insight cannot be evaluated with the available fields/mappings.")
    return "\n\n".join(report)


def main():
    frames = [read_csv(ORIGINAL_FILE), read_csv(ANONYMIZED_FILE)]
    labels = ["Original", "Anonymized"]
    out = FOLDER / OUTPUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    birth, birth_note = align_birth_years(frames)
    education, education_note = align_education(frames)
    gender = lambda s: clean(s).replace({"m": "Male", "f": "Female", "o": "Other / prefer not to say"})
    post_note = " Uses nforum_posts only. Zero is a valid count; missing, negative, noninteger and nonnumeric counts are excluded from means. CSV includes participation rates and valid denominators."
    specs = [
        ("01_birth_cohorts", "Birth-year cohort distribution", "YoB", birth, birth_note, False),
        ("02_countries", "Country representation", "cc_by_ip", clean, "Country codes are retained verbatim. NA denotes Namibia. Missing is a separate category; no locations are inferred.", False),
        ("03_education", "Educational attainment", "LoE", education, education_note, False),
        ("04_gender", "Gender composition", "gender", gender, "Missing is distinct from other / prefer not to say. Percentages include every row.", False),
        ("05_posts_education", "Forum posts by education", "LoE", education, education_note + post_note, True),
        ("06_posts_cohort", "Forum posts by birth cohort", "YoB", birth, birth_note + post_note, True),
    ]
    overview = ["# Before/after anonymization", "All figures and percentages are row-weighted. Repeated users may appear in multiple rows; these are not person-level population estimates."]
    for name, frame in zip(labels, frames):
        if "user_id" in frame:
            ids = clean(frame.user_id)
            overview.append(f"{name}: {len(frame):,} rows; {ids[ids != MISSING].nunique():,} distinct nonmissing user_id values; {ids.eq(MISSING).sum():,} missing IDs. IDs may have been transformed and are not linked across files.")
        else:
            overview.append(f"{name}: {len(frame):,} rows. Unique users cannot be counted: user_id absent.")
    removed = sorted(set(frames[0]) - set(frames[1]))
    overview.append("Removed columns: " + (", ".join(removed) or "none") + ".")
    n0, n1 = map(len, frames)
    overview.append(f"Row-count change: {n1 - n0:+,}" + (f" ({(n1 - n0) / n0:.2%})." if n0 else ". Original is empty."))
    overview.append("Counts alone do not prove which rows were suppressed. Charts compare composition, not causal effects. Arbitrary generalized labels require an explicit mapping; unavailable comparisons are flagged. This script does not certify k-anonymity.")
    overview += [draw_and_summarize(spec, frames, labels, out) for spec in specs]
    overview.append("Education code reference: https://docs.openedx.org/en/latest/developers/references/internal_data_formats/data_references/sql_schema.html#level-of-education")
    (out / "summary.md").write_text("\n\n".join(overview) + "\n", encoding="utf-8")
    print(f"Saved six charts, available aggregate tables, and summary.md to {out}")


if __name__ == "__main__":
    main()
