"""Shared, unit-testable logic for the HCAHPS sentiment analysis pipeline.

Pure functions here operate on small pandas objects and have no file I/O, so they can be
exercised directly by test_hcahps_utils.py without loading the full HCAHPS-Hospital.csv.
"""
import re

import pandas as pd

MANUAL_SENTIMENT_LABEL_MAP = {
    '"Always" quiet at night': 'positive',
    '"NO", patients would not recommend the hospital (they probably would not or definitely would not recommend it)': 'negative',
    '"Sometimes" or "never" quiet at night': 'negative',
    '"Usually" quiet at night': 'positive',
    '"YES", patients would definitely recommend the hospital': 'positive',
    '"YES", patients would probably recommend the hospital': 'positive',
    'Cleanliness - linear mean score': 'neutral',
    'Cleanliness - star rating': 'neutral',
    'Communication about medicines - linear mean score': 'neutral',
    'Communication about medicines - star rating': 'neutral',
    'Discharge information - linear mean score': 'neutral',
    'Discharge information - star rating': 'neutral',
    'Doctor communication - linear mean score': 'neutral',
    'Doctor communication - star rating': 'neutral',
    'Doctors "always" communicated well': 'positive',
    'Doctors "always" explained things so they could understand': 'positive',
    'Doctors "always" listened carefully': 'positive',
    'Doctors "always" treated them with courtesy and  respect': 'positive',
    'Doctors "sometimes" or "never" communicated well': 'negative',
    'Doctors "sometimes" or "never" explained things so they could understand': 'negative',
    'Doctors "sometimes" or "never" listened carefully': 'negative',
    'Doctors "sometimes" or "never" treated them with courtesy and  respect': 'negative',
    'Doctors "usually"  treated them with courtesy and  respect': 'positive',
    'Doctors "usually" communicated well': 'positive',
    'Doctors "usually" explained things so they could understand': 'positive',
    'Doctors "usually" listened carefully': 'positive',
    'No, staff "did not" give patients information about help after discharge': 'negative',
    'No, staff "did not" give patients information about possible symptoms': 'negative',
    'No, staff "did not" give patients this information': 'negative',
    'Nurse communication - linear mean score': 'neutral',
    'Nurse communication - star rating': 'neutral',
    'Nurses "always" communicated well': 'positive',
    'Nurses "always" explained things so they could understand': 'positive',
    'Nurses "always" listened carefully': 'positive',
    'Nurses "always" treated them with courtesy and  respect': 'positive',
    'Nurses "sometimes" or "never" communicated well': 'negative',
    'Nurses "sometimes" or "never" explained things so they could understand': 'negative',
    'Nurses "sometimes" or "never" listened carefully': 'negative',
    'Nurses "sometimes" or "never" treated them with courtesy and  respect': 'negative',
    'Nurses "usually"  treated them with courtesy and  respect': 'positive',
    'Nurses "usually" communicated well': 'positive',
    'Nurses "usually" explained things so they could understand': 'positive',
    'Nurses "usually" listened carefully': 'positive',
    'Overall hospital rating - linear mean score': 'neutral',
    'Overall hospital rating - star rating': 'neutral',
    'Patients who gave a rating of "6" or lower (low)': 'negative',
    'Patients who gave a rating of "7" or "8" (medium)': 'neutral',
    'Patients who gave a rating of "9" or "10" (high)': 'positive',
    'Quietness - linear mean score': 'neutral',
    'Quietness - star rating': 'neutral',
    'Recommend hospital - linear mean score': 'neutral',
    'Recommend hospital - star rating': 'neutral',
    'Room was "always" clean': 'positive',
    'Room was "sometimes" or "never" clean': 'negative',
    'Room was "usually" clean': 'positive',
    'Staff "always" explained': 'positive',
    'Staff "always" explained new medications': 'positive',
    'Staff "always" explained possible side effects': 'positive',
    'Staff "sometimes" or "never" explained': 'negative',
    'Staff "sometimes" or "never" explained new medications': 'negative',
    'Staff "sometimes" or "never" explained possible side effects': 'negative',
    'Staff "usually" explained': 'positive',
    'Staff "usually" explained new medications': 'positive',
    'Staff "usually" explained possible side effects': 'positive',
    'Summary star rating': 'neutral',
    'Yes, staff "did" give patients information about help after discharge': 'positive',
    'Yes, staff "did" give patients information about possible symptoms': 'positive',
    'Yes, staff "did" give patients this information': 'positive',
}

SENTIMENT_NUMERIC_MAP = {'positive': 1, 'neutral': 0, 'negative': -1}

# U.S. state, DC, and Puerto Rico population estimates: Vintage 2025 (as of July 1, 2025),
# U.S. Census Bureau, "Annual Estimates of the Resident Population" (NST-EST2025-POP),
# https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html
# Guam, U.S. Virgin Islands, American Samoa, and the Northern Mariana Islands are not part of
# the annual postcensal estimates program, so their figures are the 2020 Decennial Census
# island-area counts (released Oct 28, 2021): https://www.census.gov/data/tables/2020/dec/2020-guam.html,
# .../2020-us-virgin-islands.html, .../2020-american-samoa.html, .../2020-commonwealth-northern-mariana-islands.html
STATE_POPULATION_MAP = {
    'AL': 5193088,
    'AK': 737270,
    'AZ': 7623818,
    'AR': 3114791,
    'CA': 39355309,
    'CO': 6012561,
    'CT': 3688496,
    'DE': 1059952,
    'DC': 693645,
    'FL': 23462518,
    'GA': 11302748,
    'HI': 1432820,
    'ID': 2029733,
    'IL': 12719141,
    'IN': 6973333,
    'IA': 3238387,
    'KS': 2977220,
    'KY': 4606864,
    'LA': 4618189,
    'ME': 1414874,
    'MD': 6265347,
    'MA': 7154084,
    'MI': 10127884,
    'MN': 5830405,
    'MS': 2954160,
    'MO': 6270541,
    'MT': 1144694,
    'NE': 2018006,
    'NV': 3282188,
    'NH': 1415342,
    'NJ': 9548215,
    'NM': 2125498,
    'NY': 20002427,
    'NC': 11197968,
    'ND': 799358,
    'OH': 11900510,
    'OK': 4123288,
    'OR': 4273586,
    'PA': 13059432,
    'RI': 1114521,
    'SC': 5570274,
    'SD': 935094,
    'TN': 7315076,
    'TX': 31709821,
    'UT': 3538904,
    'VT': 644663,
    'VA': 8880107,
    'WA': 8001020,
    'WV': 1766147,
    'WI': 5972787,
    'WY': 588753,
    'PR': 3184835,
    'VI': 87146,   # 2020 Decennial Census (island-area counts don't get annual postcensal updates)
    'GU': 153836,  # 2020 Decennial Census
    'AS': 49710,   # 2020 Decennial Census
    'MP': 47329,   # 2020 Decennial Census
}

# Strips the answer-category suffix off an HCAHPS Measure ID (e.g. H_COMP_1_A_P -> H_COMP_1) so
# that "always/usually/sometimes-or-never" style answers to the *same* question are grouped
# together instead of being summed across unrelated questions.
MEASURE_SUFFIX_PATTERN = re.compile(r'_(A_P|SN_P|U_P|N_P|Y_P|DN|DY|PY|LINEAR_SCORE|STAR_RATING|0_6|7_8|9_10)$')


def derive_measure_group(measure_ids: pd.Series) -> pd.Series:
    """Collapse HCAHPS Measure IDs to their shared underlying question."""
    return measure_ids.astype(str).str.replace(MEASURE_SUFFIX_PATTERN, '', regex=True)


def compute_estimated_responses(percent: pd.Series, completed_surveys: pd.Series) -> pd.Series:
    """Approximate the number of patients behind a reported HCAHPS answer percentage.

    CMS-suppressed values ("Not Applicable"/"Not Available") should already be NaN by the time
    they reach this function; NaN inputs produce NaN outputs, which pandas .sum() skips.
    """
    return pd.to_numeric(percent, errors='coerce') * pd.to_numeric(completed_surveys, errors='coerce') / 100


def aggregate_by_state_and_measure(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate estimated responses within each (State, measure_group) -- never across groups.

    Expects columns: State, measure_group, estimated_responses, sentiment_label.
    Rows with a zero/NaN total (i.e. fully suppressed for that state+measure) are dropped.
    """
    grouped = (
        df.groupby(['State', 'measure_group'], dropna=False)
        .agg(
            total_estimated_responses=('estimated_responses', 'sum'),
            negative_estimated_responses=(
                'estimated_responses',
                lambda x: x[df.loc[x.index, 'sentiment_label'] == 'negative'].sum(),
            ),
        )
        .reset_index()
    )
    grouped = grouped[grouped['total_estimated_responses'] > 0].copy()
    grouped['negative_rate_pct'] = (
        grouped['negative_estimated_responses'] / grouped['total_estimated_responses'] * 100
    ).round(2)
    return grouped


def join_population(states: pd.Series, population_map: dict = STATE_POPULATION_MAP) -> pd.Series:
    """Look up each state's population, returning NaN (not an error) for unknown codes."""
    return states.map(population_map)


def build_recommend_hospital_summary(by_measure: pd.DataFrame, population_map: dict = STATE_POPULATION_MAP) -> pd.DataFrame:
    """Build the headline state ranking from the single 'recommend hospital' measure group."""
    summary = by_measure[by_measure['measure_group'] == 'H_RECMND'].drop(columns='measure_group').copy()
    summary = summary.rename(
        columns={
            'total_estimated_responses': 'recommend_total_estimated_responses',
            'negative_estimated_responses': 'recommend_negative_estimated_responses',
            'negative_rate_pct': 'recommend_negative_rate_pct',
        }
    )
    summary['state_population'] = join_population(summary['State'], population_map)
    summary['negative_per_million'] = (
        summary['recommend_negative_estimated_responses'] / summary['state_population'] * 1_000_000
    ).round(2)
    return summary
