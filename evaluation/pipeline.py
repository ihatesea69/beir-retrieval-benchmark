"""
Integrated Evaluation Pipeline
Combines all evaluation components into a unified interface
"""

from typing import Dict, List, Optional
import pandas as pd
import logging

from evaluation.metrics import RetrievalEvaluator
from evaluation.advanced_metrics import GradedRelevanceEvaluator, RobustMetricsCalculator, PooledEvaluator
from evaluation.precision_recall import PrecisionRecallAnalyzer
from evaluation.statistical_tests import StatisticalTester
from evaluation.reporting import EvaluationReporter

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Unified evaluation pipeline supporting binary and graded relevance,
    pooled evaluation, and statistical comparison.
    """

    def __init__(self, k_values: List[int] = [3, 5, 10, 100], graded: bool = False):
        self.k_values = k_values
        self.graded = graded

    def evaluate_system(
        self,
        results: Dict[str, List[Dict]],
        qrels: Dict[str, Dict[str, int]],
        system_name: str = "system"
    ) -> pd.DataFrame:
        """
        Evaluate a single retrieval system

        Args:
            results: {query_id: [{'id': doc_id, ...}]}
            qrels: {query_id: {doc_id: relevance}}
            system_name: Name for logging

        Returns:
            DataFrame with per-query metrics
        """
        logger.info(f"Evaluating system: {system_name}")
        return RetrievalEvaluator.evaluate_retrieval(
            results, qrels, k_values=self.k_values, graded=self.graded
        )

    def compare_systems(
        self,
        system_results: Dict[str, Dict[str, List[Dict]]],
        qrels: Dict[str, Dict[str, int]],
        metric: str = "NDCG@10"
    ) -> Dict:
        """
        Compare multiple systems with statistical significance testing

        Args:
            system_results: {system_name: results_dict}
            qrels: Ground truth relevance judgments
            metric: Metric to use for comparison

        Returns:
            Dict with per-system DataFrames and comparison table
        """
        system_dfs = {}
        for name, results in system_results.items():
            system_dfs[name] = self.evaluate_system(results, qrels, name)

        # Build per-query score lists for statistical testing
        all_query_ids = set()
        for df in system_dfs.values():
            all_query_ids.update(df['query_id'].tolist())

        system_scores = {}
        for name, df in system_dfs.items():
            if metric in df.columns:
                score_map = dict(zip(df['query_id'], df[metric]))
                system_scores[name] = [score_map.get(qid, 0.0) for qid in sorted(all_query_ids)]

        comparison_table = EvaluationReporter.generate_comparison_table(system_dfs)

        stat_comparison = None
        if len(system_scores) >= 2:
            try:
                stat_comparison = StatisticalTester.compare_systems(system_scores)
            except Exception as e:
                logger.warning(f"Statistical comparison failed: {e}")

        return {
            'system_results': system_dfs,
            'comparison_table': comparison_table,
            'statistical_comparison': stat_comparison
        }

    def pr_analysis(
        self,
        results: Dict[str, List[Dict]],
        qrels: Dict[str, Dict[str, int]]
    ) -> Dict:
        """
        Compute PR curves and R-Precision for all queries

        Returns:
            Dict with per-query PRCurveData and averaged 11-point curve
        """
        pr_data = {}
        all_interpolated = []

        for query_id, docs in results.items():
            if query_id not in qrels:
                continue
            retrieved = [d['id'] for d in docs]
            relevant = [d for d, s in qrels[query_id].items() if s > 0]

            curve = PrecisionRecallAnalyzer.analyze_query(query_id, retrieved, relevant)
            pr_data[query_id] = curve
            all_interpolated.append(curve.interpolated_11_point)

        averaged = PrecisionRecallAnalyzer.average_interpolated_pr(all_interpolated)

        return {'per_query': pr_data, 'averaged_11_point': averaged}
