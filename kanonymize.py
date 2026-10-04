"""Compare the actual raw and anonymized CSV files beside this script.
Install: python -m pip install pandas numpy matplotlib
Run:     python compare_anonymization.py

Designed for: city/postalCode deletion, YYYY -> YYY*, and education mapped to
'college or above', 'no college education', and 'Unknown'. No suppression rate
is assumed. All results count rows, not distinct people.
"""
from pathlib import Path
import textwrap
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FOLDER = Path(__file__).resolve().parent
RAW_FILE = 'raw_data.csv'
ANON_FILE = 'anonymized_data.csv'
OUTPUT_FOLDER = 'outputs/comparison_v2'
TOP_COUNTRIES = 12
UNKNOWN = 'Unknown'
EDU_ORDER = ['college or above', 'no college education', UNKNOWN]
COLLEGE = {'a', 'b', 'm', 'p', 'p_se', 'p_oth'}
SCHOOL = {'hs', 'jhs', 'el', 'none'}
EDU_LABELS = {'a': 'Associate', 'b': 'Bachelor', 'm': 'Master / professional',
              'p': 'Doctorate', 'p_se': 'Doctorate: science/engineering',
              'p_oth': 'Doctorate: other', 'hs': 'High school', 'jhs': 'Junior high',
              'el': 'Elementary', 'none': 'No formal education', 'other': 'Other education'}
NOT_RECORDED = '[Missing]'
INVALID = '[Unrecognized birth year]'


def clean(series):
    # NA is Namibia. Education 'none' means no formal schooling.
    s = series.fillna('').astype(str).str.strip()
    return s.mask(s.str.lower().isin({'', 'null', 'nan', '<na>'}), NOT_RECORDED)


def education(series):
    def recode(v):
        x = v.lower()
        if x in COLLEGE or x == 'college or above':
            return EDU_ORDER[0]
        if x in SCHOOL or x == 'no college education':
            return EDU_ORDER[1]
        if x in {'other', 'unknown'} or v == NOT_RECORDED:
            return UNKNOWN
        raise ValueError(f'Unrecognized LoE value {v!r}. Add its meaning to education().')
    return clean(series).map(recode)


def decades(series):
    def recode(v):
        if v == NOT_RECORDED or v.lower() == 'unknown':
            return UNKNOWN
        if re.fullmatch(r'\d{4}', v) or re.fullmatch(r'\d{3}\*', v):
            return v[:3] + '*'
        return INVALID
    return clean(series).map(recode)


def posts(series):
    s = pd.to_numeric(clean(series), errors='coerce')
    return s.where(np.isfinite(s) & s.ge(0) & s.mod(1).eq(0))


def unavailable(frame, columns):
    absent = [c for c in columns if c not in frame]
    return 'Analysis unavailable: ' + ', '.join(absent) + ' removed' if absent else None


def distribution(series, categories):
    counts = series.value_counts().reindex(categories, fill_value=0)
    return pd.DataFrame({'category': categories, 'rows': counts.values,
                         'percent': counts.values / len(series) * 100 if len(series) else np.nan})


def activity(frame, category, categories):
    work = pd.DataFrame({'category': category, 'posts': posts(frame.nforum_posts)})
    grouped = work.groupby('category').posts
    valid = grouped.count().reindex(categories, fill_value=0)
    active = work.assign(active=work.posts.gt(0).where(work.posts.notna()))
    rate = active.groupby('category').active.mean().reindex(categories) * 100
    return pd.DataFrame({'category': categories,
                         'rows': work.category.value_counts().reindex(categories, fill_value=0).values,
                         'valid_post_rows': valid.values,
                         'percent_posting': rate.values,
                         'mean_posts': grouped.mean().reindex(categories).values})


def bars(ax, table, metric, color, limit=None):
    y = np.arange(len(table))
    values = table[metric].to_numpy(dtype=float)
    ax.barh(y, np.nan_to_num(values), color=color)
    ax.set_yticks(y, table.category)
    ax.invert_yaxis()
    finite = values[np.isfinite(values)]
    maximum = max(float(finite.max()) if len(finite) else 1, .1)
    ax.set_xlim(0, limit if limit is not None else maximum * 1.43)
    for i, row in table.iterrows():
        v = row[metric]
        n = row['valid_post_rows'] if 'valid_post_rows' in table else row['rows']
        label = f'{v:.1f}%; n={int(n):,}' if pd.notna(v) else 'No valid observations'
        ax.text((v if pd.notna(v) else 0) + ax.get_xlim()[1] * .012, i,
                label, va='center', fontsize=8)
    ax.set_xlabel('% with ≥1 forum post among valid counts' if metric == 'percent_posting' else '% of all rows')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)


def blocked(ax, reason):
    ax.text(.5, .5, textwrap.fill(reason, 40), transform=ax.transAxes,
            ha='center', va='center', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, path, title, note):
    fig.suptitle(title, fontsize=16)
    # Reserve enough room for the explanatory caption, including long caveats.
    fig.text(.02, .015, textwrap.fill(note, 150), fontsize=9, va='bottom')
    fig.tight_layout(rect=(0, .16, 1, .94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def difference(before, after, metric):
    joined = before.set_index('category')[[metric]].join(
        after.set_index('category')[[metric]], lsuffix='_raw', rsuffix='_anon')
    delta = joined[metric + '_anon'] - joined[metric + '_raw']
    delta = delta.dropna()
    if delta.empty:
        return 'No jointly observable categories; comparison unavailable.'
    biggest = delta.abs().idxmax()
    if delta.abs().max() < 1e-10:
        return ('Broad-category results are numerically identical. This does NOT mean no information '
                'was lost: distinctions inside merged categories are no longer recoverable.')
    return (f'Largest displayed change: {biggest}, {delta[biggest]:+.2f} percentage points. '
            'This is a descriptive change, not a statistical-significance or causal claim.')


def comparison(raw, anon, out, slug, title, field, transform, note, is_activity=False, categories=None):
    required = [field] + (['nforum_posts'] if is_activity else [])
    frames = [raw, anon]
    reasons = [unavailable(f, required) for f in frames]
    series = [None if reason else transform(f[field]) for f, reason in zip(frames, reasons)]
    if categories is None:
        categories = sorted(set().union(*(set(s) for s in series if s is not None)))
    tables = [None if s is None else (activity(f, s, categories) if is_activity else distribution(s, categories))
              for f, s in zip(frames, series)]
    metric = 'percent_posting' if is_activity else 'percent'
    maxima = [t[metric].max() for t in tables if t is not None and t[metric].notna().any()]
    limit = max(maxima + [.1]) * 1.43
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(categories) * .38 + 3)))
    exported = []
    for ax, label, f, reason, table, color in zip(axes, ['Original at matching granularity', 'Anonymized'], frames,
                                                reasons, tables, ['#3676a8', '#d08038']):
        ax.set_title(f'{label}\n{len(f):,} rows')
        if reason:
            blocked(ax, reason)
        else:
            bars(ax, table, metric, color, limit)
            exported.append(table.assign(dataset=label))
    save(fig, out / f'{slug}.png', title, note)
    if exported:
        pd.concat(exported, ignore_index=True).to_csv(out / f'{slug}.csv', index=False)
    result = [f'## {title}', note]
    for label, table, reason in zip(['Original', 'Anonymized'], tables, reasons):
        if reason:
            result.append(f'{label}: {reason}.')
        elif table[metric].notna().any():
            top = table.loc[table[metric].idxmax()]
            result.append(f'{label}: highest displayed category is {top.category} ({top[metric]:.2f}%).')
        else:
            result.append(f'{label}: no usable observations.')
    result.append(difference(*tables, metric) if all(t is not None for t in tables)
                  else 'The original insight cannot be compared because a required field was removed.')
    return '\n\n'.join(result)


def detail_loss(raw, anon, out):
    """Keep raw detail visible rather than quietly coarsening both sides."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    sections = []
    for row, field, label, transform in [(0, 'LoE', 'Education', education), (1, 'YoB', 'Birth year', decades)]:
        reason = unavailable(raw, [field])
        if reason:
            for ax in axes[row]:
                blocked(ax, reason)
            sections.append(reason)
            continue
        values = clean(raw[field])
        if field == 'LoE':
            labels = values.replace(EDU_LABELS)
            cats = labels.value_counts().index.tolist()
            table = distribution(labels, cats)
            bars(axes[row, 0], table, 'percent', '#3676a8')
            known = table[~table.category.isin([NOT_RECORDED, UNKNOWN])]
            if not known.empty:
                top = known.loc[known.percent.idxmax()]
                sections.append(f'Original education insight: the most frequent reported detailed category is '
                                f'{top.category}, {top.percent:.2f}% of all original rows. '
                                'With only the two college categories, the most frequent individual degree '
                                'can no longer be determined from the anonymized data.')
        else:
            years = pd.to_numeric(values.where(values.str.fullmatch(r'\d{4}')), errors='coerce')
            frequencies = years.dropna().astype(int).value_counts().sort_index()
            axes[row, 0].plot(frequencies.index, frequencies.values / len(raw) * 100, color='#3676a8')
            axes[row, 0].set_xlabel('Exact birth year (missing / malformed values excluded from line)')
            axes[row, 0].set_ylabel('% of all original rows')
            table = pd.DataFrame({'birth_year': frequencies.index, 'rows': frequencies.values})
            if not frequencies.empty:
                peak = int(frequencies.idxmax())
                sections.append(f'Original birth-year insight: {peak} is a most frequent exact reported birth year '
                                f'({int(frequencies.max()):,} rows). A decade mask cannot identify which year '
                                'within a decade is most frequent. The largest decade can still be compared.')
        table.to_csv(out / f'original_detail_{field}.csv', index=False)
        axes[row, 0].set_title(f'Original {label.lower()}: available detail')
        axes[row, 1].set_title(f'Anonymized {label.lower()}: remaining detail')
        reason = unavailable(anon, [field])
        if reason:
            blocked(axes[row, 1], reason)
            sections.append(reason)
        else:
            grouped = transform(anon[field])
            categories = EDU_ORDER if field == 'LoE' else sorted(grouped.unique())
            table = distribution(grouped, categories)
            bars(axes[row, 1], table, 'percent', '#d08038')
            sections.append(f'{label}: {values.nunique():,} original distinct labels -> '
                            f'{clean(anon[field]).nunique():,} distinct labels actually stored after anonymization '
                            '(including missing/unknown).')
    note = ('These panels intentionally show DIFFERENT resolutions. Use the separate matched-category charts '
            'for numerical before/after comparisons. Original degree levels and exact birth years are visible '
            'here because their loss would be hidden if both panels were generalized first.')
    save(fig, out / '01_detail_lost.png', 'What generalization removes', note)
    return '## Detail lost through generalization\n\n' + '\n\n'.join(sections + [note])


def main():
    out = FOLDER / OUTPUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(FOLDER / RAW_FILE, dtype=str, keep_default_na=False)
    anon = pd.read_csv(FOLDER / ANON_FILE, dtype=str, keep_default_na=False)
    if raw.empty:
        raise ValueError('raw_data.csv has no rows to analyze.')
    # Refuse unexpected birth-band formats rather than claiming they are decades.
    for name, frame in [('Original', raw), ('Anonymized', anon)]:
        if 'YoB' in frame:
            unsupported = clean(frame.YoB).str.contains(r'\d\s*[-–—]\s*\d', regex=True)
            if unsupported.any():
                raise ValueError(f'{name} contains birth-year ranges. This version expects exact years or 199* decade masks.')
    report = ['# Anonymization: information loss and changes in composition',
              'All charts are row-weighted, not person-weighted. Missing values remain explicit. '
              'Country NA is Namibia; zero forum posts is an observation, not missing data. '
              'The label “no college education” means no reported college degree, not proof of never attending college.']
    report.append(f'Original rows: {len(raw):,}. Anonymized rows: {len(anon):,}. '
                  f'Net row-count reduction: {len(raw)-len(anon):,} ({(len(raw)-len(anon))/len(raw):.2%}). '
                  'This is calculated from your files; 2.73% is not hard-coded. '
                  'A count difference alone does not prove suppression or identify the removed records.')
    for name, f in [('Original', raw), ('Anonymized', anon)]:
        if 'user_id' in f:
            ids = clean(f.user_id)
            report.append(f'{name}: {ids[ids != NOT_RECORDED].nunique():,} distinct nonmissing user IDs. '
                          'Repeated IDs contribute multiple rows to these charts.')
    report.append('Removed columns: ' + ', '.join(sorted(set(raw) - set(anon))) + '.')
    for c in ['city', 'postalCode']:
        if c in raw and c not in anon:
            report.append(f'Analysis unavailable: {c} removed. Within-country geographic patterns cannot be recovered.')
    qi = [c for c in ['cc_by_ip', 'city', 'postalCode', 'LoE', 'YoB', 'gender'] if c in anon]
    if len(anon) and qi:
        sizes = anon.groupby(qi, dropna=False).size()
        report.append(f'Actual row-level minimum k across {qi}: {int(sizes.min())}. '
                      f'Rows still in groups smaller than 5: {int(sizes[sizes < 5].sum()):,}. '
                      'This is a limited QI check, not a guarantee against other released fields.')
    if len(raw) == len(anon):
        report.append('Both files have equal row counts. Generalization alone can leave all broad-category '
                      'distributions identical. Suppression has not been demonstrated by the row counts.')
    report.append(detail_loss(raw, anon, out))
    edu_note = ('Both datasets are explicitly mapped to college or above, no college education, and Unknown. '
                'Differences here show changes in broad composition; distinctions between associate, bachelor, '
                'master and doctorate are no longer available in the anonymized data.')
    decade_note = ('Raw exact years are grouped to decades ONLY for this matched comparison. '
                   '1991 and 199* both map to 199*. Identical shapes are expected from masking alone; '
                   'the separate detail-loss figure shows why exact-year insights are still lost. '
                   'These are birth cohorts, not ages at course enrollment.')
    report.append(comparison(raw, anon, out, '02_education_composition', 'Did the education mix change?',
                             'LoE', education, edu_note, categories=EDU_ORDER))
    report.append(comparison(raw, anon, out, '03_cohort_composition', 'Which birth cohorts changed in representation?',
                             'YoB', decades, decade_note))
    available = [clean(f.cc_by_ip) for f in [raw, anon] if 'cc_by_ip' in f]
    # Select top countries from ORIGINAL data when possible and freeze that list.
    top = (available[0][available[0] != NOT_RECORDED].value_counts().head(TOP_COUNTRIES).index.tolist()
           if available else [])
    country_categories = top + ['Other countries', NOT_RECORDED]
    country = lambda s: clean(s).map(lambda x: x if x in top or x == NOT_RECORDED else 'Other countries')
    report.append(comparison(raw, anon, out, '04_country_composition', 'Did country representation change?',
                             'cc_by_ip', country,
                             'Country-level geography survives only if cc_by_ip remains. The same country list is used '
                             'in both panels; other countries are pooled. City and postal-code insights cannot be '
                             'reconstructed when those columns are deleted.', categories=country_categories))
    post_note = (' Participation means at least one nforum_posts entry, among rows with a valid nonnegative '
                 'integer count. Missing counts are excluded, never set to zero. Labels show valid sample sizes. '
                 'CSV tables also include mean posts and total category rows.')
    report.append(comparison(raw, anon, out, '05_participation_education',
                             'Does the education–forum participation relationship survive?',
                             'LoE', education, edu_note + post_note, True, EDU_ORDER))
    report.append(comparison(raw, anon, out, '06_participation_cohort',
                             'Does the birth-cohort–forum participation relationship survive?',
                             'YoB', decades, decade_note + post_note, True))
    report.append('## Interpreting the results\n\nThe matching-resolution charts isolate changes in aggregate '
                  'composition from the mechanical relabeling of categories. They cannot establish causality '
                  'or prove that retained rows are an unbiased sample. The detail-loss figure shows what no '
                  'matching-resolution chart can recover: degree distinctions and exact birth years. '
                  'No exact ages, missing demographics, or suppressed records are inferred.')
    (out / 'summary.md').write_text('\n\n'.join(report) + '\n', encoding='utf-8')
    print(f'Saved six figures, aggregate CSVs, and summary.md to {out}')


if __name__ == '__main__':
    main()
