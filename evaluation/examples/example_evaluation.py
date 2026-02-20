"""
Example: IR Evaluation Enhancement Usage
Demonstrates graded relevance, PR curves, statistical testing, and TREC format
"""

from evaluation.pipeline import EvaluationPipeline
from evaluation.advanced_metrics import GradedRelevanceEvaluator
from evaluation.precision_recall import PrecisionRecallAnalyzer
from evaluation.statistical_tests import StatisticalTester
from evaluation.reporting import EvaluationReporter

# ── 1. Graded Relevance Evaluation ────────────────────────────────────────────
qrels = {
    'q1': {'d1': 4, 'd2': 2, 'd3': 0, 'd4': 1},
    'q2': {'d5': 3, 'd6': 0, 'd7': 2},
}

results_a = {
    'q1': [{'id': 'd1', 'score': 0.9}, {'id': 'd2', 'score': 0.8},
           {'id': 'd3', 'score': 0.5}, {'id': 'd4', 'score': 0.3}],
    'q2': [{'id': 'd5', 'score': 0.9}, {'id': 'd7', 'score': 0.7}, {'id': 'd6', 'score': 0.2}],
}

results_b = {
    'q1': [{'id': 'd3', 'score': 0.9}, {'id': 'd1', 'score': 0.6},
           {'id': 'd4', 'score': 0.4}, {'id': 'd2', 'score': 0.2}],
    'q2': [{'id': 'd6', 'score': 0.9}, {'id': 'd5', 'score': 0.5}, {'id': 'd7', 'score': 0.3}],
}

# Evaluate with graded relevance
pipeline = EvaluationPipeline(k_values=[3, 10], graded=True)
df_a = pipeline.evaluate_system(results_a, qrels, "SystemA")
print("System A results:")
print(df_a[['query_id', 'NDCG@10', 'GradedNDCG@10', 'MAP']].to_string())

# ── 2. Precision-Recall Curves ────────────────────────────────────────────────
retrieved = ['d1', 'd3', 'd2', 'd5', 'd4']
relevant = ['d1', 'd2', 'd4']

pr_points = PrecisionRecallAnalyzer.compute_pr_points(retrieved, relevant)
interpolated = PrecisionRecallAnalyzer.interpolate_11_point(pr_points)
r_prec = PrecisionRecallAnalyzer.calculate_r_precision(retrieved, relevant)
f1 = PrecisionRecallAnalyzer.calculate_f_measure(0.6, 0.5)

print(f"\nR-Precision: {r_prec:.3f}")
print(f"F1: {f1:.3f}")
print("11-point interpolated PR:")
for level, prec in sorted(interpolated.items()):
    print(f"  R={level:.1f}: P={prec:.3f}")

# ── 3. Statistical Comparison ─────────────────────────────────────────────────
scores_a = [0.8, 0.7, 0.9, 0.75, 0.85]
scores_b = [0.6, 0.5, 0.7, 0.55, 0.65]

result = StatisticalTester.paired_t_test(scores_a, scores_b)
effect = StatisticalTester.effect_size_cohens_d(scores_a, scores_b)
print(f"\nStatistical test: {StatisticalTester.interpret_significance(result['p_value'])}")
print(f"Effect size: {StatisticalTester.interpret_effect_size(effect)} (d={effect:.2f})")

# ── 4. System Comparison with Reporting ───────────────────────────────────────
comparison = pipeline.compare_systems(
    {'SystemA': results_a, 'SystemB': results_b}, qrels
)
print("\nSystem comparison:")
print(comparison['comparison_table'].to_string())
