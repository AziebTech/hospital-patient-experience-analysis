import pandas as pd

from hcahps_utils import (
    MANUAL_SENTIMENT_LABEL_MAP,
    SENTIMENT_NUMERIC_MAP,
    STATE_POPULATION_MAP,
    aggregate_by_state_and_measure,
    build_recommend_hospital_summary,
    compute_estimated_responses,
    derive_measure_group,
)

# Load the HCAHPS dataset
file_path = 'HCAHPS-Hospital.csv'
df = pd.read_csv(file_path, dtype=str)

# Data provenance: confirm this run is operating on the real, public CMS HCAHPS Hospital
# dataset (not synthetic/example data) and record exactly what was loaded.
print('=' * 70)
print('DATA PROVENANCE')
print('=' * 70)
print(f'Source file: {file_path}')
print('Source: U.S. Centers for Medicare & Medicaid Services (CMS), public HCAHPS Hospital survey data')
print(f'Total rows loaded: {len(df):,}')
if 'Facility ID' in df.columns:
    print(f'Unique facilities: {df["Facility ID"].nunique():,}')
if 'State' in df.columns:
    print(f'Unique states/territories: {df["State"].nunique()}')
if 'Start Date' in df.columns and 'End Date' in df.columns:
    print(f'Reporting period: {df["Start Date"].min()} to {df["End Date"].max()}')
print('=' * 70)

feature_column = 'HCAHPS Answer Description'
if feature_column not in df.columns:
    raise ValueError(f'Expected column "{feature_column}" not found in dataset.')

# Map labels to rows and add a numeric sentiment column
manual_sentiment_label_map = MANUAL_SENTIMENT_LABEL_MAP
sentiment_label_map = SENTIMENT_NUMERIC_MAP
state_population_map = STATE_POPULATION_MAP

df['sentiment_label'] = df[feature_column].map(manual_sentiment_label_map)
df['sentiment_numeric'] = df['sentiment_label'].map(sentiment_label_map)
df['HCAHPS Answer Percent Numeric'] = pd.to_numeric(df['HCAHPS Answer Percent'], errors='coerce')
df['Number of Completed Surveys Numeric'] = pd.to_numeric(df['Number of Completed Surveys'], errors='coerce')
df['estimated_responses'] = compute_estimated_responses(
    df['HCAHPS Answer Percent Numeric'], df['Number of Completed Surveys Numeric']
)
df['measure_group'] = derive_measure_group(df['HCAHPS Measure ID'])

if 'State' in df.columns:
    df['State'] = df['State'].astype(str).str.strip()
    df['state_population'] = df['State'].map(state_population_map)
else:
    df['state_population'] = pd.NA

# Report mapping coverage and example samples
mapped_count = df['sentiment_label'].notna().sum()
missing_count = df['sentiment_label'].isna().sum()
unique_missing = df.loc[df['sentiment_label'].isna(), feature_column].unique()

print(f'Total rows: {len(df)}')
print(f'Mapped sentiment rows: {mapped_count}')
print(f'Missing sentiment rows: {missing_count}')

if missing_count > 0:
    print('\nUnmapped feature values:')
    for value in sorted(unique_missing):
        print(f' - {value}')

print('\nSample rows with sentiment mapping:')
print(df[[feature_column, 'sentiment_label', 'sentiment_numeric']].head(20).to_string(index=False))

if 'State' in df.columns:
    df['State'] = df['State'].astype(str).str.strip()

    # Correct, per-question aggregation: sum estimated responses within each (State, measure_group)
    # so different HCAHPS questions are never blended into a single count/rate.
    by_measure = aggregate_by_state_and_measure(df)

    by_measure_output_path = 'HCAHPS-Hospital-state-negative-by-measure.csv'
    by_measure.sort_values(['measure_group', 'negative_rate_pct'], ascending=[True, False]).to_csv(
        by_measure_output_path, index=False
    )
    print(f'\nSaved per-question state negative-rate breakdown to: {by_measure_output_path}')

    # Headline state comparison: use ONLY the "Recommend hospital" question (H_RECMND). It is a
    # single, mutually-exclusive, apples-to-apples measure of overall sentiment per facility, so
    # it can be summed across facilities within a state without blending unrelated questions or
    # double-counting the same patients across multiple HCAHPS composites.
    negative_by_state = build_recommend_hospital_summary(by_measure, state_population_map)
    missing_population = negative_by_state.loc[negative_by_state['state_population'].isna(), 'State'].tolist()

    # Coverage disclosure: the recommend-hospital ranking only includes facilities/states that
    # have a reportable (non CMS-suppressed) percentage for that specific measure. Surface exactly
    # how much of the national dataset that represents, rather than leaving it implicit.
    all_states = set(df['State'].unique())
    ranked_states = set(negative_by_state['State'])
    states_with_no_recmnd_data = sorted(all_states - ranked_states)
    total_facilities = df['Facility ID'].nunique()
    recmnd_rows = df[df['measure_group'] == 'H_RECMND']
    facilities_with_recmnd_data = recmnd_rows.loc[recmnd_rows['estimated_responses'].notna(), 'Facility ID'].nunique()

    print('\nCoverage of the "recommend hospital" headline ranking:')
    print(f'  States/territories in raw data: {len(all_states)}')
    print(f'  States/territories represented in ranking: {len(ranked_states)}')
    if states_with_no_recmnd_data:
        print(f'  Excluded (no reportable recommend-hospital data): {states_with_no_recmnd_data}')
    print(f'  Facilities in raw data: {total_facilities:,}')
    print(f'  Facilities with reportable recommend-hospital data: {facilities_with_recmnd_data:,}')
    print(f'  Facilities excluded (CMS-suppressed for this measure): {total_facilities - facilities_with_recmnd_data:,}')
    print(f'  Total patients represented in ranking: {negative_by_state["recommend_total_estimated_responses"].sum():,.0f}')

    print('\nRecommend-hospital negative sentiment counts by State (descending):')
    print(negative_by_state.sort_values('recommend_negative_estimated_responses', ascending=False).to_string(index=False))

    print('\nRecommend-hospital negative sentiment rate by State (% of estimated responses):')
    print(negative_by_state.sort_values('recommend_negative_rate_pct', ascending=False).to_string(index=False))

    if missing_population:
        print('\nWarning: state population missing for:')
        for state in missing_population:
            print(f' - {state}')

    if negative_by_state['state_population'].notna().sum() >= 2:
        corr = negative_by_state[['recommend_negative_estimated_responses', 'state_population']].corr().loc[
            'recommend_negative_estimated_responses', 'state_population'
        ]
        print(f'\nCorrelation between recommend-hospital negative estimated responses and state population: {corr:.4f}')

    # Validation: compare our derived metric against CMS's OWN official "Summary star rating"
    # (H_STAR_RATING) for the same states. This is the correct sanity check -- unlike third-party
    # composite indices (e.g. WalletHub-style "best/worst healthcare" rankings, which blend cost,
    # access, and public-health variables unrelated to HCAHPS) -- because both numbers come from
    # the same underlying CMS survey and should move together if our pipeline is sound.
    official_star = df[df['HCAHPS Measure ID'] == 'H_STAR_RATING'].copy()
    official_star['Patient Survey Star Rating Numeric'] = pd.to_numeric(
        official_star['Patient Survey Star Rating'], errors='coerce'
    )
    state_official_star = (
        official_star.groupby('State')['Patient Survey Star Rating Numeric'].mean().reset_index()
        .rename(columns={'Patient Survey Star Rating Numeric': 'avg_cms_summary_star_rating'})
    )
    negative_by_state = negative_by_state.merge(state_official_star, on='State', how='left')
    validation_corr = negative_by_state[['recommend_negative_rate_pct', 'avg_cms_summary_star_rating']].corr().loc[
        'recommend_negative_rate_pct', 'avg_cms_summary_star_rating'
    ]
    print(
        f'\nValidation: correlation between our recommend-hospital negative rate and CMS\'s own '
        f'official Summary Star Rating by state: {validation_corr:.3f} (expected to be strongly negative)'
    )

    summary_output_path = 'HCAHPS-Hospital-state-negative-summary.csv'
    negative_by_state.sort_values('recommend_negative_rate_pct', ascending=False).to_csv(summary_output_path, index=False)
    print(f'\nSaved state summary to: {summary_output_path}')
else:
    print('\nState column not found, skipping negative sentiment by state count.')

# Save a copy with sentiment labels if desired
output_path = 'HCAHPS-Hospital-sentiment.csv'
df.to_csv(output_path, index=False)
print(f'\nSaved mapped dataset to: {output_path}')
