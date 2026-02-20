"""
Unit Tests for IR Evaluation Enhancement
Covers graded relevance, PR curves, statistical tests, TREC format,
robust metrics, visualization, and reporting.
"""

import pytest
import math
import os
import tempfile
import pandas as pd

from evaluation.advanced_metrics import GradedRelevanceEvaluator, RobustMetricsCalculator
from evaluation.precision_recall import PrecisionRecallAnalyzer, STANDARD_RECALL_LEVELS
from evaluation.statistical_tests import StatisticalTester
from evaluation.trec_format import TRECFormatHandler
from evaluation.reporting import EvaluationReporter


# ─── Graded Relevance ──────────────────────────────────────────────────────────

class TestGradedRelevanceEvaluator:

    def test_graded_ndcg_uses_actual_scores(self):
        """Graded NDCG should differ from binary when high-relevance docs are ranked lower"""
        # d2 has high relevance (4) but is ranked 2nd; d1 has low relevance (1) ranked 1st
        retrieved = ['d1', 'd2', 'd3']
        qrels_graded = {'d1': 1, 'd2': 4, 'd3': 0}
        qrels_binary = {'d1': 1, 'd2': 1, 'd3': 0}

        graded = GradedRelevanceEvaluator.calculate_graded_ndcg(retrieved, qrels_graded, k=3)
        binary = GradedRelevanceEvaluator.calculate_graded_ndcg(retrieved, qrels_binary, k=3)

        # Graded NDCG < 1.0 because high-relevance doc is not ranked first
        # Binary NDCG = 1.0 because both relevant docs are in top-2
        assert graded < 1.0
        assert binary == 1.0

    def test_graded_ndcg_perfect_ranking(self):
        """Perfect ranking should give NDCG = 1.0"""
        qrels = {'d1': 4, 'd2': 2, 'd3': 1}
        retrieved = ['d1', 'd2', 'd3']
        ndcg = GradedRelevanceEvaluator.calculate_graded_ndcg(retrieved, qrels, k=3)
        assert abs(ndcg - 1.0) < 1e-9

    def test_graded_ndcg_empty_inputs(self):
        assert GradedRelevanceEvaluator.calculate_graded_ndcg([], {}, k=10) == 0.0
        assert GradedRelevanceEvaluator.calculate_graded_ndcg(['d1'], {}, k=10) == 0.0

    def test_convert_to_binary_threshold(self):
        qrels = {'d1': 0, 'd2': 1, 'd3': 2, 'd4': 3}
        binary = GradedRelevanceEvaluator.convert_to_binary(qrels, threshold=2)
        assert binary == {'d1': 0, 'd2': 0, 'd3': 1, 'd4': 1}

    def test_validate_relevance_scale_valid(self):
        qrels = {'d1': 0, 'd2': 2, 'd3': 4}
        assert GradedRelevanceEvaluator.validate_relevance_scale(qrels) is True

    def test_validate_relevance_scale_invalid(self):
        qrels = {'d1': 5}  # 5 is out of 0-4 range
        assert GradedRelevanceEvaluator.validate_relevance_scale(qrels) is False

    def test_all_zero_relevance(self):
        qrels = {'d1': 0, 'd2': 0}
        ndcg = GradedRelevanceEvaluator.calculate_graded_ndcg(['d1', 'd2'], qrels, k=2)
        assert ndcg == 0.0


# ─── Precision-Recall Curves ──────────────────────────────────────────────────

class TestPrecisionRecallAnalyzer:

    def test_pr_points_example1_from_slides(self):
        """Reproduce Example 1 from IR evaluation slides"""
        # Docs 588, 589, 590, 592, 772 are relevant (5 found), total=6
        retrieved = ['588', '589', '576', '590', '986', '592', '984', '988', '578', '985',
                     '103', '591', '772', '990']
        relevant = ['588', '589', '590', '592', '772', 'missing_doc']

        pr_points = PrecisionRecallAnalyzer.compute_pr_points(retrieved, relevant)

        # Should have 5 points (missing_doc never retrieved)
        assert len(pr_points) == 5

        # First point: rank 1, 1 relevant found
        assert abs(pr_points[0][0] - 1/6) < 1e-9  # recall
        assert abs(pr_points[0][1] - 1.0) < 1e-9  # precision

    def test_pr_points_example2_from_slides(self):
        """Reproduce Example 2 from IR evaluation slides"""
        retrieved = ['588', '576', '589', '342', '590', '717', '984', '772', '321', '498',
                     '113', '628', '772b', '592']
        relevant = ['588', '589', '590', '772', '321', '592']

        pr_points = PrecisionRecallAnalyzer.compute_pr_points(retrieved, relevant)
        assert len(pr_points) == 6
        # Last point should reach recall=1.0
        assert abs(pr_points[-1][0] - 1.0) < 1e-9

    def test_11_point_interpolation_has_11_levels(self):
        pr_points = [(0.2, 0.8), (0.5, 0.6), (1.0, 0.4)]
        interpolated = PrecisionRecallAnalyzer.interpolate_11_point(pr_points)
        assert len(interpolated) == 11
        assert all(level in interpolated for level in STANDARD_RECALL_LEVELS)

    def test_interpolation_max_precision_rule(self):
        pr_points = [(0.3, 0.9), (0.6, 0.5), (1.0, 0.3)]
        interpolated = PrecisionRecallAnalyzer.interpolate_11_point(pr_points)
        # At recall=0.0, max precision at any recall >= 0.0 = 0.9
        assert abs(interpolated[0.0] - 0.9) < 1e-9
        # At recall=0.4, max precision at recall >= 0.4 = 0.5
        assert abs(interpolated[0.4] - 0.5) < 1e-9

    def test_r_precision_example_from_slides(self):
        """R=6 relevant docs, 4 found in top-6 → R-Precision = 4/6"""
        retrieved = ['588', '589', '576', '590', '986', '592', '984', '988', '578', '985',
                     '103', '591', '772', '990']
        relevant = ['588', '589', '590', '592', '772', 'missing_doc']
        r_prec = PrecisionRecallAnalyzer.calculate_r_precision(retrieved, relevant)
        assert abs(r_prec - 4/6) < 1e-9

    def test_f_measure_standard(self):
        f1 = PrecisionRecallAnalyzer.calculate_f_measure(0.7, 0.5, beta=1.0)
        expected = 2 * 0.7 * 0.5 / (0.7 + 0.5)
        assert abs(f1 - expected) < 1e-9

    def test_f_measure_zero_inputs(self):
        assert PrecisionRecallAnalyzer.calculate_f_measure(0.0, 0.0) == 0.0
        assert PrecisionRecallAnalyzer.calculate_f_measure(0.0, 0.5) == 0.0

    def test_f_measure_beta_weighting(self):
        """beta > 1 should weight recall more → higher score when recall > precision"""
        p, r = 0.3, 0.8
        f_recall_weighted = PrecisionRecallAnalyzer.calculate_f_measure(p, r, beta=2.0)
        f_precision_weighted = PrecisionRecallAnalyzer.calculate_f_measure(p, r, beta=0.5)
        assert f_recall_weighted > f_precision_weighted

    def test_average_interpolated_pr(self):
        curves = [
            {0.0: 1.0, 0.1: 0.9, 0.2: 0.8, 0.3: 0.7, 0.4: 0.6,
             0.5: 0.5, 0.6: 0.4, 0.7: 0.3, 0.8: 0.2, 0.9: 0.1, 1.0: 0.0},
            {0.0: 0.8, 0.1: 0.7, 0.2: 0.6, 0.3: 0.5, 0.4: 0.4,
             0.5: 0.3, 0.6: 0.2, 0.7: 0.1, 0.8: 0.0, 0.9: 0.0, 1.0: 0.0},
        ]
        avg = PrecisionRecallAnalyzer.average_interpolated_pr(curves)
        assert abs(avg[0.0] - 0.9) < 1e-9
        assert abs(avg[1.0] - 0.0) < 1e-9


# ─── Statistical Tests ────────────────────────────────────────────────────────

class TestStatisticalTester:

    def test_paired_t_test_returns_required_keys(self):
        a = [0.5, 0.6, 0.7, 0.8, 0.9]
        b = [0.4, 0.5, 0.6, 0.7, 0.8]
        result = StatisticalTester.paired_t_test(a, b)
        assert all(k in result for k in ['t_statistic', 'p_value', 'mean_difference'])

    def test_paired_t_test_significant_difference(self):
        """Clearly different systems should yield low p-value"""
        a = [0.9] * 20
        b = [0.1] * 20
        result = StatisticalTester.paired_t_test(a, b)
        assert result['p_value'] < 0.001

    def test_confidence_interval_contains_mean(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.4, 0.55, 0.65, 0.75, 0.85]
        lo, hi = StatisticalTester.compute_confidence_interval(scores)
        mean = sum(scores) / len(scores)
        assert lo <= mean <= hi

    def test_bonferroni_correction_caps_at_1(self):
        p_values = [0.8, 0.9, 0.95]
        corrected = StatisticalTester.bonferroni_correction(p_values)
        assert all(v <= 1.0 for v in corrected)

    def test_bonferroni_empty(self):
        assert StatisticalTester.bonferroni_correction([]) == []

    def test_cohens_d_zero_for_identical(self):
        scores = [0.5, 0.6, 0.7, 0.8]
        d = StatisticalTester.effect_size_cohens_d(scores, scores)
        assert abs(d) < 1e-9

    def test_significance_interpretation(self):
        assert "***" in StatisticalTester.interpret_significance(0.0001)
        assert "**" in StatisticalTester.interpret_significance(0.005)
        assert "*" in StatisticalTester.interpret_significance(0.03)
        assert "ns" in StatisticalTester.interpret_significance(0.1)


# ─── TREC Format ──────────────────────────────────────────────────────────────

class TestTRECFormatHandler:

    def _write_tmp(self, content: str, suffix: str = '.qrels') -> str:
        f = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8')
        f.write(content)
        f.close()
        return f.name

    def test_load_qrels_basic(self):
        content = "1 0 doc1 2\n1 0 doc2 0\n2 0 doc3 1\n"
        path = self._write_tmp(content)
        try:
            qrels = TRECFormatHandler.load_qrels(path)
            assert qrels['1']['doc1'] == 2
            assert qrels['1']['doc2'] == 0
            assert qrels['2']['doc3'] == 1
        finally:
            os.unlink(path)

    def test_round_trip_qrels(self):
        original = {'1': {'docA': 2, 'docB': 0}, '2': {'docC': 1}}
        with tempfile.NamedTemporaryFile(suffix='.qrels', delete=False) as f:
            path = f.name
        try:
            TRECFormatHandler.save_qrels(original, path)
            loaded = TRECFormatHandler.load_qrels(path)
            assert loaded == original
        finally:
            os.unlink(path)

    def test_load_results_basic(self):
        content = "1 Q0 doc1 1 0.9 run1\n1 Q0 doc2 2 0.8 run1\n"
        path = self._write_tmp(content, suffix='.run')
        try:
            results = TRECFormatHandler.load_results(path)
            assert len(results['1']) == 2
            assert results['1'][0]['id'] == 'doc1'
            assert results['1'][0]['rank'] == 1
        finally:
            os.unlink(path)

    def test_validate_qrels_detects_wrong_field_count(self):
        content = "1 doc1 2\n"  # Missing iteration field
        path = self._write_tmp(content)
        try:
            errors = TRECFormatHandler.validate_qrels_format(path)
            assert len(errors) > 0
        finally:
            os.unlink(path)

    def test_validate_qrels_detects_invalid_relevance(self):
        content = "1 0 doc1 5\n"  # 5 is out of range
        path = self._write_tmp(content)
        try:
            errors = TRECFormatHandler.validate_qrels_format(path)
            assert len(errors) > 0
        finally:
            os.unlink(path)

    def test_load_qrels_skips_comments(self):
        content = "# comment\n1 0 doc1 1\n"
        path = self._write_tmp(content)
        try:
            qrels = TRECFormatHandler.load_qrels(path)
            assert '1' in qrels
        finally:
            os.unlink(path)


# ─── Robust Metrics ───────────────────────────────────────────────────────────

class TestRobustMetricsCalculator:

    def test_bpref_all_relevant_judged(self):
        """When all judged docs are relevant, bpref = 1.0"""
        retrieved = ['d1', 'd2', 'd3']
        relevant = ['d1', 'd2', 'd3']
        judged = {'d1', 'd2', 'd3'}
        bpref = RobustMetricsCalculator.calculate_bpref(retrieved, relevant, judged)
        assert bpref == 1.0

    def test_bpref_no_relevant(self):
        bpref = RobustMetricsCalculator.calculate_bpref(['d1', 'd2'], [], {'d1', 'd2'})
        assert bpref == 0.0

    def test_bpref_range(self):
        retrieved = ['d1', 'd2', 'd3', 'd4', 'd5']
        relevant = ['d1', 'd3']
        judged = {'d1', 'd2', 'd3', 'd4', 'd5'}
        bpref = RobustMetricsCalculator.calculate_bpref(retrieved, relevant, judged)
        assert 0.0 <= bpref <= 1.0

    def test_inf_ap_basic(self):
        retrieved = ['d1', 'd2', 'd3']
        qrels = {'d1': 1, 'd2': 0, 'd3': 1}
        judged = {'d1', 'd2', 'd3'}
        inf_ap = RobustMetricsCalculator.calculate_inf_ap(retrieved, qrels, judged)
        assert 0.0 <= inf_ap <= 1.0

    def test_inf_ap_empty(self):
        assert RobustMetricsCalculator.calculate_inf_ap([], {}, set()) == 0.0

    def test_estimate_recall_base(self):
        estimate = RobustMetricsCalculator.estimate_recall_base(10, 100, 1000)
        assert estimate >= 10  # At least as many as found


# ─── Reporting ────────────────────────────────────────────────────────────────

class TestEvaluationReporter:

    def _sample_df(self):
        return pd.DataFrame([
            {'query_id': 'q1', 'NDCG@10': 0.8, 'MAP': 0.7},
            {'query_id': 'q2', 'NDCG@10': 0.6, 'MAP': 0.5},
            {'query_id': 'q3', 'NDCG@10': 0.9, 'MAP': 0.85},
        ])

    def test_summary_table_contains_all_metrics(self):
        df = self._sample_df()
        summary = EvaluationReporter.generate_summary_table(df, "TestSystem")
        assert 'NDCG@10' in summary.index
        assert 'MAP' in summary.index
        assert 'mean' in summary.columns

    def test_identify_best_worst_queries(self):
        df = self._sample_df()
        result = EvaluationReporter.identify_best_worst_queries(df, 'NDCG@10', n=1)
        assert result['best'].iloc[0]['query_id'] == 'q3'
        assert result['worst'].iloc[0]['query_id'] == 'q2'

    def test_comparison_table_has_system_columns(self):
        df1 = self._sample_df()
        df2 = self._sample_df().assign(**{'NDCG@10': [0.5, 0.4, 0.6], 'MAP': [0.4, 0.3, 0.5]})
        comparison = EvaluationReporter.generate_comparison_table({'A': df1, 'B': df2})
        assert 'A' in comparison.columns
        assert 'B' in comparison.columns

    def test_export_csv(self):
        df = self._sample_df()
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            path = f.name
        try:
            EvaluationReporter.export_csv(df, path)
            loaded = pd.read_csv(path)
            assert 'NDCG@10' in loaded.columns
        finally:
            os.unlink(path)

    def test_export_json(self):
        import json
        data = {'metric': 'NDCG@10', 'value': 0.75}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            EvaluationReporter.export_json(data, path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['metric'] == 'NDCG@10'
        finally:
            os.unlink(path)

    def test_export_latex(self):
        df = self._sample_df().set_index('query_id')
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as f:
            path = f.name
        try:
            EvaluationReporter.export_latex(df, path)
            with open(path) as f:
                content = f.read()
            assert 'tabular' in content
        finally:
            os.unlink(path)
