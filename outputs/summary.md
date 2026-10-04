# Before/after anonymization

All figures and percentages are row-weighted. Repeated users may appear in multiple rows; these are not person-level population estimates.

Original: 199,999 rows; 188,034 distinct nonmissing user_id values; 0 missing IDs. IDs may have been transformed and are not linked across files.

Anonymized: 194,530 rows; 182,846 distinct nonmissing user_id values; 0 missing IDs. IDs may have been transformed and are not linked across files.

Removed columns: city, postalCode.

Row-count change: -5,469 (-2.73%).

Counts alone do not prove which rows were suppressed. Charts compare composition, not causal effects. Arbitrary generalized labels require an explicit mapping; unavailable comparisons are flagged. This script does not certify k-anonymity.

## Birth-year cohort distribution

Birth-year cohorts, not ages. Common bins never split an input band. Exact years are displayed at decade resolution unless broader input bands require further merging. Missing and unrecognized values remain separate. Generalization prevents recovering within-band differences.

Original: highest displayed category is 1980–1989: 29.77 % of rows.

Anonymized: highest displayed category is 1980–1989: 30.14 % of rows.

Largest absolute change among jointly observable categories: 1990–1999, +0.38 percentage points.

Displayed composition is broadly preserved under a descriptive 2-percentage-point threshold.

## Country representation

Country codes are retained verbatim. NA denotes Namibia. Missing is a separate category; no locations are inferred. Less frequent categories are pooled identically in both panels, selected from combined row counts.

Original: highest displayed category is US: 29.80 % of rows.

Anonymized: highest displayed category is US: 30.60 % of rows.

Largest absolute change among jointly observable categories: [Other categories], -1.56 percentage points.

Displayed composition is broadly preserved under a descriptive 2-percentage-point threshold.

## Educational attainment

Original education retains detailed categories; anonymized education shows college or above, no college education, and Unknown.

Categories have different granularity, so direct category-by-category percentage changes are not calculated.

## Gender composition

Missing is distinct from other / prefer not to say. Percentages include every row.

Original: highest displayed category is Male: 51.20 % of rows.

Anonymized: highest displayed category is Male: 51.42 % of rows.

Largest absolute change among jointly observable categories: Male, +0.23 percentage points.

Displayed composition is broadly preserved under a descriptive 2-percentage-point threshold.

## Forum posts by education

Original education retains detailed categories; anonymized education shows college or above, no college education, and Unknown. Uses nforum_posts only. Zero is a valid count; missing, negative, noninteger and nonnumeric counts are excluded from means. CSV includes participation rates and valid denominators.

Categories have different granularity, so direct category-by-category percentage changes are not calculated.

## Forum posts by birth cohort

Birth-year cohorts, not ages. Common bins never split an input band. Exact years are displayed at decade resolution unless broader input bands require further merging. Missing and unrecognized values remain separate. Generalization prevents recovering within-band differences. Uses nforum_posts only. Zero is a valid count; missing, negative, noninteger and nonnumeric counts are excluded from means. CSV includes participation rates and valid denominators.

Original: highest displayed category is 1910–1919: 7.47 mean posts.

Anonymized: highest displayed category is 1890–1899: 3.12 mean posts.

Largest absolute change among jointly observable categories: 1910–1919, -7.47 posts per row.

Insight remains evaluable at this granularity; inspect mean changes alongside valid sample sizes. No statistical significance is claimed.

Education code reference: https://docs.openedx.org/en/latest/developers/references/internal_data_formats/data_references/sql_schema.html#level-of-education
