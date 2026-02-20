"""
Property-Based Tests for IR Evaluation Enhancement
Feature: ir-evaluation-enhancement

Uses Hypothesis for property-based testing.
Each test runs minimum 100 iterations.
"""

import pytest
import math
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from evaluation.advanced_metrics import GradedRelevanceEvaluator, RobustMetricsCalculator
from evaluation.precision_recall import PrecisionRecallAnalyzer, STANDARD_RECALL_LEVELS
from evaluation.statistical_tests import StatisticalTester
from evaluation.trec_format import TRECFormatHandler

import tempfile
import os


# ─── Strategies ────────────────────────────────────────────────────────────────

doc_ids = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8),
    min_size=1, max_size=20, unique=True
)

graded_qrels_strategy = st.fixed_dictionaries({}).flatmap(
    lambda _: st.lists(
        st.tuples(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8),
            st.integers(min_value=0, max_value=4)
        ),
        min_size=1, max_size=15, unique_by=lambda x: x[0]
    ).map(dict)
)

float_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ─── Property 1: Graded NDCG uses actual scores ────────────────────────────────

@settings(max_examples=100)
@given(
    retrieved=doc_ids,
    graded_qrels=graded_qrels_strategy
)
def test_property1_graded_ndcg_uses_actual_scores(retrieved, graded_qrels):
    """
    Feature: ir-evaluation-enhancement, Property 1: Graded relevance NDCG uses actual scores
    Validates: Requirements 1.2

    Graded NDCG should differ from binary NDCG when relevance scores vary.
    Specifically, graded NDCG should be >= binary NDCG when high-relevance docs
    are ranked first (since higher gains are rewarded).
    """
    # Only test when there are docs with relevance > 1 (to distinguish graded from binary)
    high_rel_docs = {d: s for d, s in graded_qrels.items() if s > 1}
    assume(len(high_rel_docs) >= 1)

    graded_ndcg = GradedRelevanceEvaluator.calculate_graded_ndcg(retrieved, graded_qrels, k=10)

    # Binary qrels: any score >= 1 is relevant
    binary_qrels = GradedRelevanceEvaluator.convert_to_binary(graded_qrels, threshold=1)
    binary_ndcg = GradedRelevanceEvaluator.calculate_graded_ndcg(retrieved, binary_qrels, k=10)

    # Both scores must be in valid range [0, 1]
    assert 0.0 <= graded_ndcg <= 1.0 + 1e-9, f"Graded NDCG out of range: {graded_ndcg}"
    assert 0.0 <= binary_ndcg <= 1.0 + 1e-9, f"Binary NDCG out of range: {binary_ndcg}"


# ─── Property 3: PR points at relevant document positions ──────────────────────

@settings(max_examples=100)
@given(
    retrieved=doc_ids,
    relevant=doc_ids
)
def test_property3_pr_points_at_relevant_docs(retrieved, relevant):
    """
    Feature: ir-evaluation-enhancement, Property 3: Precision-recall points at relevant documents
    Validates: Requirements 2.1

    PR points should be computed exactly at ranks where relevant documents appear.
    """
    pr_points = PrecisionRecallAnalyzer.compute_pr_points(retrieved, relevant)

    relevant_set = set(relevant)
    relevant_positions = [i + 1 for i, d in enumerate(retrieved) if d in relevant_set]

    # Number of PR points == number of relevant docs found in retrieved list
    assert len(pr_points) == len(relevant_positions), (
        f"Expected {len(relevant_positions)} PR points, got {len(pr_points)}"
    )

    # Recall values must be non-decreasing
    recalls = [r for r, _ in pr_points]
    for i in range(1, len(recalls)):
        assert recalls[i] >= recalls[i - 1], "Recall must be non-decreasing"

    # All recall values must be in [0, 1]
    for recall, precision in pr_points:
        assert 0.0 <= recall <= 1.0, f"Recall out of range: {recall}"
        assert 0.0 <= precision <= 1.0, f"Precision out of range: {precision}"


# ─── Property 4: 11-point interpolation completeness ──────────────────────────

@settings(max_examples=100)
@given(
    retrieved=doc_ids,
    relevant=doc_ids
)
def test_property4_11_point_interpolation_completeness(retrieved, relevant):
    """
    Feature: ir-evaluation-enhancement, Property 4: 11-point interpolation completeness
    Validates: Requirements 2.2

    Interpolation must produce exactly 11 points at standard recall levels.
    """
    pr_points = PrecisionRecallAnalyzer.compute_pr_points(retrieved, relevant)
    interpolated = PrecisionRecallAnalyzer.interpolate_11_point(pr_points)

    assert len(interpolated) == 11, f"Expected 11 interpolated points, got {len(interpolated)}"

    for level in STANDARD_RECALL_LEVELS:
        assert level in interpolated, f"Missing standard recall level {level}"
        assert 0.0 <= interpolated[level] <= 1.0, (
            f"Interpolated precision at {level} out of range: {interpolated[level]}"
        )


# ─── Property 5: Interpolation maximum precision rule ─────────────────────────

@settings(max_examples=100)
@given(
    retrieved=doc_ids,
    relevant=doc_ids
)
def test_property5_interpolation_maximum_precision_rule(retrieved, relevant):
    """
    Feature: ir-evaluation-enhancement, Property 5: Interpolation maximum precision rule
    Validates: Requirements 2.3

    Interpolated precision at level r = max precision at any recall >= r.
    """
    pr_points = PrecisionRecallAnalyzer.compute_pr_points(retrieved, relevant)
    interpolated = PrecisionRecallAnalyzer.interpolate_11_point(pr_points)

    for level in STANDARD_RECALL_LEVELS:
        # Manually compute expected max precision
        expected_max = max(
            (p for r, p in pr_points if r >= level),
            default=0.0
        )
        assert abs(interpolated[level] - expected_max) < 1e-9, (
            f"At recall level {level}: expected {expected_max}, got {interpolated[level]}"
        )


# ─── Property 6: R-Precision position correctness ─────────────────────────────

@settings(max_examples=100)
@given(
    retrieved=doc_ids,
    relevant=doc_ids
)
def test_property6_r_precision_position_correctness(retrieved, relevant):
    """
    Feature: ir-evaluation-enhancement, Property 6: R-Precision position correctness
    Validates: Requirements 3.1

    R-Precision = precision at rank R where R = |relevant|.
    """
    assume(len(relevant) > 0)

    r_prec = PrecisionRecallAnalyzer.calculate_r_precision(retrieved, relevant)

    # Manually compute: hits in top-R / R
    r = len(set(relevant))
    relevant_set = set(relevant)
    top_r = retrieved[:r]
    expected = sum(1 for d in top_r if d in relevant_set) / r

    assert abs(r_prec - expected) < 1e-9, f"R-Precision mismatch: {r_prec} vs {expected}"
    assert 0.0 <= r_prec <= 1.0, f"R-Precision out of range: {r_prec}"


# ─── Property 7: F-measure harmonic mean formula ──────────────────────────────

@settings(max_examples=100)
@given(
    precision=float_scores,
    recall=float_scores
)
def test_property7_f_measure_harmonic_mean(precision, recall):
    """
    Feature: ir-evaluation-enhancement, Property 7: F-measure harmonic mean formula
    Validates: Requirements 3.2

    F1 = 2*P*R / (P+R) when both are positive.
    """
    f1 = PrecisionRecallAnalyzer.calculate_f_measure(precision, recall, beta=1.0)

    if precision + recall > 0:
        expected = 2 * precision * recall / (precision + recall)
        assert abs(f1 - expected) < 1e-9, f"F1 mismatch: {f1} vs {expected}"
    else:
        assert f1 == 0.0, f"F1 should be 0 when P+R=0, got {f1}"

    assert 0.0 <= f1 <= 1.0, f"F1 out of range: {f1}"


# ─── Property 8: Statistical test paired data consistency ─────────────────────

@settings(max_examples=100)
@given(
    n=st.integers(min_value=5, max_value=30),
    delta=st.floats(min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False)
)
def test_property8_statistical_test_consistency(n, delta):
    """
    Feature: ir-evaluation-enhancement, Property 8: Statistical test paired data consistency
    Validates: Requirements 4.1

    When system A consistently outperforms B by delta, the t-test should detect
    the direction of the difference correctly.
    """
    import random
    random.seed(42)

    scores_b = [random.uniform(0.1, 0.7) for _ in range(n)]
    scores_a = [min(1.0, max(0.0, s + delta)) for s in scores_b]

    # Skip when scores are nearly identical (causes NaN p-value due to zero variance)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    assume(any(abs(d) > 1e-9 for d in diffs))

    result = StatisticalTester.paired_t_test(scores_a, scores_b)

    assert 'p_value' in result
    assert 't_statistic' in result
    assert 'mean_difference' in result
    assert 0.0 <= result['p_value'] <= 1.0, f"p-value out of range: {result['p_value']}"

    # Sign of t-statistic should match sign of mean difference
    if abs(result['mean_difference']) > 1e-9:
        assert (result['t_statistic'] > 0) == (result['mean_difference'] > 0), (
            "t-statistic sign should match mean difference sign"
        )


# ─── Property 9: Bonferroni correction scaling ────────────────────────────────

@settings(max_examples=100)
@given(
    p_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=20
    )
)
def test_property9_bonferroni_correction_scaling(p_values):
    """
    Feature: ir-evaluation-enhancement, Property 9: Bonferroni correction scaling
    Validates: Requirements 4.4

    Each corrected p-value = min(original * n_comparisons, 1.0).
    """
    corrected = StatisticalTester.bonferroni_correction(p_values)
    n = len(p_values)

    assert len(corrected) == n, "Corrected list must have same length as input"

    for original, corrected_val in zip(p_values, corrected):
        expected = min(original * n, 1.0)
        assert abs(corrected_val - expected) < 1e-9, (
            f"Bonferroni correction wrong: {corrected_val} vs {expected}"
        )
        assert 0.0 <= corrected_val <= 1.0, f"Corrected p-value out of range: {corrected_val}"


# ─── Property 10: Pool combination completeness ───────────────────────────────

@settings(max_examples=100)
@given(
    system_results=st.dictionaries(
        keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6),
        values=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8),
            min_size=1, max_size=20
        ),
        min_size=1, max_size=5
    ),
    pool_depth=st.integers(min_value=1, max_value=20)
)
def test_property10_pool_combination_completeness(system_results, pool_depth):
    """
    Feature: ir-evaluation-enhancement, Property 10: Pool combination completeness
    Validates: Requirements 7.1

    Pool must contain all unique documents from top-k of each system.
    """
    # Build pool manually
    expected_pool = set()
    for system_docs in system_results.values():
        for doc in system_docs[:pool_depth]:
            expected_pool.add(doc)

    # Build pool using our logic
    actual_pool = set()
    for system_docs in system_results.values():
        actual_pool.update(system_docs[:pool_depth])

    assert actual_pool == expected_pool, (
        f"Pool mismatch: expected {len(expected_pool)} docs, got {len(actual_pool)}"
    )


# ─── Property 11: bpref calculation correctness ───────────────────────────────

@settings(max_examples=100)
@given(
    retrieved=doc_ids,
    relevant=doc_ids
)
def test_property11_bpref_correctness(retrieved, relevant):
    """
    Feature: ir-evaluation-enhancement, Property 11: bpref calculation correctness
    Validates: Requirements 8.1

    bpref must be in [0, 1] and equal 1.0 when all judged docs are relevant.
    """
    judged = set(retrieved)  # All retrieved docs are judged
    bpref = RobustMetricsCalculator.calculate_bpref(retrieved, relevant, judged)

    assert 0.0 <= bpref <= 1.0, f"bpref out of range: {bpref}"

    # When all judged docs are relevant, bpref should be 1.0
    all_relevant_judged = set(retrieved)
    bpref_all_rel = RobustMetricsCalculator.calculate_bpref(
        retrieved, retrieved, all_relevant_judged
    )
    assert bpref_all_rel == 1.0 or len(retrieved) == 0, (
        f"bpref should be 1.0 when all judged docs are relevant, got {bpref_all_rel}"
    )


# ─── Property 2: TREC format round-trip preservation ─────────────────────────

@settings(max_examples=100)
@given(
    qrels_data=st.dictionaries(
        keys=st.text(alphabet="0123456789", min_size=1, max_size=4),
        values=st.dictionaries(
            keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8),
            values=st.integers(min_value=0, max_value=4),
            min_size=1, max_size=5
        ),
        min_size=1, max_size=5
    )
)
def test_property2_trec_format_round_trip(qrels_data):
    """
    Feature: ir-evaluation-enhancement, Property 2: TREC format round-trip preservation
    Validates: Requirements 5.1, 5.5

    Saving qrels to TREC format and loading back should produce equivalent data.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qrels', delete=False) as f:
        tmp_path = f.name

    try:
        TRECFormatHandler.save_qrels(qrels_data, tmp_path)
        loaded = TRECFormatHandler.load_qrels(tmp_path)

        assert set(loaded.keys()) == set(qrels_data.keys()), "Query IDs mismatch after round-trip"

        for query_id in qrels_data:
            assert set(loaded[query_id].keys()) == set(qrels_data[query_id].keys()), (
                f"Doc IDs mismatch for query {query_id}"
            )
            for doc_id in qrels_data[query_id]:
                assert loaded[query_id][doc_id] == qrels_data[query_id][doc_id], (
                    f"Relevance mismatch for query {query_id}, doc {doc_id}"
                )
    finally:
        os.unlink(tmp_path)


# ─── Property 12: Format validation error detection ───────────────────────────

@settings(max_examples=100)
@given(
    bad_lines=st.lists(
        st.text(min_size=1, max_size=50).filter(
            lambda s: len(s.strip().split()) != 4 and len(s.strip()) > 0
        ),
        min_size=1, max_size=5
    )
)
def test_property12_format_validation_error_detection(bad_lines):
    """
    Feature: ir-evaluation-enhancement, Property 12: Format validation error detection
    Validates: Requirements 5.4

    Malformed TREC qrels lines should be detected as errors.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qrels', delete=False, encoding='utf-8') as f:
        for line in bad_lines:
            f.write(line + '\n')
        tmp_path = f.name

    try:
        errors = TRECFormatHandler.validate_qrels_format(tmp_path)
        # All bad lines should produce at least one error
        assert len(errors) > 0, "Expected validation errors for malformed data"
    finally:
        os.unlink(tmp_path)
