"""Automated tests for the shared HCAHPS pipeline logic in hcahps_utils.py.

Run with: python -m unittest test_hcahps_utils.py -v

These tests use small synthetic DataFrames (not the 100MB+ HCAHPS-Hospital.csv), so they run
fast and don't depend on the raw data file being present.
"""
import math
import unittest

import pandas as pd

from hcahps_utils import (
    aggregate_by_state_and_measure,
    build_recommend_hospital_summary,
    compute_estimated_responses,
    derive_measure_group,
    join_population,
)


class TestDeriveMeasureGroup(unittest.TestCase):
    def test_percentage_category_suffixes_collapse_to_shared_root(self):
        ids = pd.Series(['H_COMP_1_A_P', 'H_COMP_1_SN_P', 'H_COMP_1_U_P'])
        self.assertTrue((derive_measure_group(ids) == 'H_COMP_1').all())

    def test_linear_score_and_star_rating_share_root_with_percentage_rows(self):
        ids = pd.Series(['H_COMP_1_A_P', 'H_COMP_1_LINEAR_SCORE', 'H_COMP_1_STAR_RATING'])
        self.assertTrue((derive_measure_group(ids) == 'H_COMP_1').all())

    def test_recommend_hospital_yes_no_suffixes_collapse_together(self):
        ids = pd.Series(['H_RECMND_DN', 'H_RECMND_DY', 'H_RECMND_PY', 'H_RECMND_LINEAR_SCORE'])
        self.assertTrue((derive_measure_group(ids) == 'H_RECMND').all())

    def test_hospital_rating_bucket_suffixes_collapse_together(self):
        ids = pd.Series(['H_HSP_RATING_0_6', 'H_HSP_RATING_7_8', 'H_HSP_RATING_9_10'])
        self.assertTrue((derive_measure_group(ids) == 'H_HSP_RATING').all())

    def test_unrelated_questions_remain_distinct(self):
        ids = pd.Series(['H_COMP_1_A_P', 'H_COMP_2_A_P', 'H_RECMND_DY'])
        result = derive_measure_group(ids)
        self.assertEqual(result.nunique(), 3)


class TestComputeEstimatedResponses(unittest.TestCase):
    def test_basic_multiplication(self):
        result = compute_estimated_responses(pd.Series(['50']), pd.Series(['200']))
        self.assertAlmostEqual(result.iloc[0], 100.0)

    def test_suppressed_values_become_nan_not_zero(self):
        result = compute_estimated_responses(pd.Series(['Not Available']), pd.Series(['200']))
        self.assertTrue(math.isnan(result.iloc[0]))

    def test_missing_survey_count_becomes_nan(self):
        result = compute_estimated_responses(pd.Series(['75']), pd.Series(['Not Applicable']))
        self.assertTrue(math.isnan(result.iloc[0]))


class TestAggregateByStateAndMeasure(unittest.TestCase):
    def _make_two_question_facility(self):
        """One facility (952 completed surveys) answering two distinct HCAHPS questions.

        This mirrors the real dataset's structure: every question for a facility shares the
        same completed-surveys base, split into always/usually/sometimes-or-never rows.
        """
        rows = [
            # Question 1 (nurse communication): 76% always (positive), 6% sometimes/never (negative), 18% usually (positive)
            {'State': 'FL', 'measure_group': 'H_COMP_1', 'estimated_responses': 76 * 952 / 100, 'sentiment_label': 'positive'},
            {'State': 'FL', 'measure_group': 'H_COMP_1', 'estimated_responses': 6 * 952 / 100, 'sentiment_label': 'negative'},
            {'State': 'FL', 'measure_group': 'H_COMP_1', 'estimated_responses': 18 * 952 / 100, 'sentiment_label': 'positive'},
            # Question 2 (recommend hospital): 70% yes-definitely (positive), 20% yes-probably (positive), 10% no (negative)
            {'State': 'FL', 'measure_group': 'H_RECMND', 'estimated_responses': 70 * 952 / 100, 'sentiment_label': 'positive'},
            {'State': 'FL', 'measure_group': 'H_RECMND', 'estimated_responses': 20 * 952 / 100, 'sentiment_label': 'positive'},
            {'State': 'FL', 'measure_group': 'H_RECMND', 'estimated_responses': 10 * 952 / 100, 'sentiment_label': 'negative'},
        ]
        return pd.DataFrame(rows)

    def test_each_measure_group_total_equals_facility_survey_count_not_the_sum_of_all_questions(self):
        df = self._make_two_question_facility()
        result = aggregate_by_state_and_measure(df)

        comp1_total = result.loc[result['measure_group'] == 'H_COMP_1', 'total_estimated_responses'].iloc[0]
        recmnd_total = result.loc[result['measure_group'] == 'H_RECMND', 'total_estimated_responses'].iloc[0]

        # Regression guard: each question's total should be ~952 (the facility's real survey
        # count), not ~1904 (952 blended across both questions), which was the original bug.
        self.assertAlmostEqual(comp1_total, 952.0, delta=0.01)
        self.assertAlmostEqual(recmnd_total, 952.0, delta=0.01)

    def test_negative_rate_calculated_per_measure_group(self):
        df = self._make_two_question_facility()
        result = aggregate_by_state_and_measure(df)

        comp1_rate = result.loc[result['measure_group'] == 'H_COMP_1', 'negative_rate_pct'].iloc[0]
        recmnd_rate = result.loc[result['measure_group'] == 'H_RECMND', 'negative_rate_pct'].iloc[0]

        self.assertAlmostEqual(comp1_rate, 6.0, delta=0.1)
        self.assertAlmostEqual(recmnd_rate, 10.0, delta=0.1)

    def test_zero_total_rows_are_excluded(self):
        df = pd.DataFrame([
            {'State': 'WY', 'measure_group': 'H_RECMND', 'estimated_responses': float('nan'), 'sentiment_label': 'negative'},
        ])
        result = aggregate_by_state_and_measure(df)
        self.assertTrue(result.empty)

    def test_different_states_are_not_mixed_together(self):
        df = pd.DataFrame([
            {'State': 'FL', 'measure_group': 'H_RECMND', 'estimated_responses': 100.0, 'sentiment_label': 'negative'},
            {'State': 'CA', 'measure_group': 'H_RECMND', 'estimated_responses': 50.0, 'sentiment_label': 'positive'},
        ])
        result = aggregate_by_state_and_measure(df)
        self.assertEqual(len(result), 2)
        fl_total = result.loc[result['State'] == 'FL', 'total_estimated_responses'].iloc[0]
        ca_total = result.loc[result['State'] == 'CA', 'total_estimated_responses'].iloc[0]
        self.assertAlmostEqual(fl_total, 100.0)
        self.assertAlmostEqual(ca_total, 50.0)


class TestPopulationJoin(unittest.TestCase):
    def test_known_state_maps_to_expected_population(self):
        result = join_population(pd.Series(['CA']), {'CA': 39355309})
        self.assertEqual(result.iloc[0], 39355309)

    def test_unknown_state_returns_nan_not_a_key_error(self):
        result = join_population(pd.Series(['ZZ']), {'CA': 39355309})
        self.assertTrue(result.isna().iloc[0])

    def test_all_56_states_and_territories_present_in_default_map(self):
        from hcahps_utils import STATE_POPULATION_MAP
        self.assertEqual(len(STATE_POPULATION_MAP), 56)
        for code in ('AS', 'GU', 'MP', 'VI', 'PR', 'DC'):
            self.assertIn(code, STATE_POPULATION_MAP)


class TestBuildRecommendHospitalSummary(unittest.TestCase):
    def test_filters_to_recommend_hospital_measure_only(self):
        by_measure = pd.DataFrame([
            {'State': 'FL', 'measure_group': 'H_RECMND', 'total_estimated_responses': 200.0,
             'negative_estimated_responses': 20.0, 'negative_rate_pct': 10.0},
            {'State': 'FL', 'measure_group': 'H_COMP_1', 'total_estimated_responses': 200.0,
             'negative_estimated_responses': 40.0, 'negative_rate_pct': 20.0},
        ])
        result = build_recommend_hospital_summary(by_measure, {'FL': 1000})
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['recommend_negative_rate_pct'], 10.0)

    def test_negative_per_million_computed_correctly(self):
        by_measure = pd.DataFrame([
            {'State': 'FL', 'measure_group': 'H_RECMND', 'total_estimated_responses': 200.0,
             'negative_estimated_responses': 20.0, 'negative_rate_pct': 10.0},
        ])
        result = build_recommend_hospital_summary(by_measure, {'FL': 2_000_000})
        self.assertAlmostEqual(result.iloc[0]['negative_per_million'], 10.0)

    def test_state_missing_from_population_map_yields_nan_per_million_not_a_crash(self):
        by_measure = pd.DataFrame([
            {'State': 'ZZ', 'measure_group': 'H_RECMND', 'total_estimated_responses': 200.0,
             'negative_estimated_responses': 20.0, 'negative_rate_pct': 10.0},
        ])
        result = build_recommend_hospital_summary(by_measure, {'FL': 2_000_000})
        self.assertTrue(math.isnan(result.iloc[0]['negative_per_million']))


if __name__ == '__main__':
    unittest.main()
