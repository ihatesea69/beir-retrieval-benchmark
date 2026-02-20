"""
Precision-Recall Analysis
Implements interpolated PR curves, R-Precision, and F-measure variants
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

STANDARD_RECALL_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@dataclass
class PRCurveData:
    """Data structure for precision-recall curve"""
    query_id: str
    pr_points: List[Tuple[float, float]]       # (recall, precision) at each relevant doc
    interpolated_11_point: Dict[float, float]  # recall_level -> interpolated precision
    r_precision: float
    auc: float


class PrecisionRecallAnalyzer:
    """
    Analyzer for precision-recall curves and related metrics
    """

    @staticmethod
    def compute_pr_points(
        retrieved: List[str],
        relevant: List[str]
    ) -> List[Tuple[float, float]]:
        """
        Compute (recall, precision) pairs at each rank where a relevant doc appears

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: List of relevant document IDs

        Returns:
            List of (recall, precision) tuples at relevant document positions
        """
        if not relevant:
            return []

        relevant_set = set(relevant)
        total_relevant = len(relevant_set)
        pr_points = []
        hits = 0

        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                hits += 1
                precision = hits / i
                recall = hits / total_relevant
                pr_points.append((recall, precision))

        return pr_points

    @staticmethod
    def interpolate_11_point(
        pr_points: List[Tuple[float, float]]
    ) -> Dict[float, float]:
        """
        Compute 11-point interpolated precision-recall curve

        At each standard recall level, precision = max precision at any recall >= that level

        Args:
            pr_points: List of (recall, precision) tuples

        Returns:
            Dict mapping standard recall levels to interpolated precision values
        """
        interpolated = {}

        for recall_level in STANDARD_RECALL_LEVELS:
            # Max precision at any recall >= recall_level
            max_precision = 0.0
            for recall, precision in pr_points:
                if recall >= recall_level:
                    max_precision = max(max_precision, precision)
            interpolated[recall_level] = max_precision

        return interpolated

    @staticmethod
    def calculate_r_precision(
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """
        Calculate R-Precision: precision at the R-th rank where R = |relevant|

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: List of relevant document IDs

        Returns:
            R-Precision score
        """
        if not relevant:
            return 0.0

        r = len(set(relevant))
        retrieved_r = retrieved[:r]
        relevant_set = set(relevant)

        hits = sum(1 for doc_id in retrieved_r if doc_id in relevant_set)
        return hits / r

    @staticmethod
    def calculate_f_measure(
        precision: float,
        recall: float,
        beta: float = 1.0
    ) -> float:
        """
        Calculate F-measure (harmonic mean of precision and recall)

        F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
        When beta=1: standard F1 = 2*P*R / (P+R)

        Args:
            precision: Precision value
            recall: Recall value
            beta: Beta parameter (>1 weights recall more, <1 weights precision more)

        Returns:
            F-measure score
        """
        if precision + recall == 0:
            return 0.0

        beta_sq = beta ** 2
        return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

    @staticmethod
    def average_interpolated_pr(
        all_interpolated: List[Dict[float, float]]
    ) -> Dict[float, float]:
        """
        Average interpolated PR curves across multiple queries

        Args:
            all_interpolated: List of 11-point interpolated dicts per query

        Returns:
            Dict of mean precision at each standard recall level
        """
        if not all_interpolated:
            return {level: 0.0 for level in STANDARD_RECALL_LEVELS}

        averaged = {}
        for level in STANDARD_RECALL_LEVELS:
            values = [interp.get(level, 0.0) for interp in all_interpolated]
            averaged[level] = float(np.mean(values))

        return averaged

    @staticmethod
    def compute_auc(pr_points: List[Tuple[float, float]]) -> float:
        """
        Compute area under the PR curve using trapezoidal rule

        Args:
            pr_points: List of (recall, precision) tuples

        Returns:
            AUC-PR score
        """
        if len(pr_points) < 2:
            return 0.0

        sorted_points = sorted(pr_points, key=lambda x: x[0])
        recalls = [p[0] for p in sorted_points]
        precisions = [p[1] for p in sorted_points]

        return float(np.trapezoid(precisions, recalls))

    @classmethod
    def analyze_query(
        cls,
        query_id: str,
        retrieved: List[str],
        relevant: List[str]
    ) -> PRCurveData:
        """
        Full PR analysis for a single query

        Args:
            query_id: Query identifier
            retrieved: Ranked retrieved document IDs
            relevant: Relevant document IDs

        Returns:
            PRCurveData with all computed values
        """
        pr_points = cls.compute_pr_points(retrieved, relevant)
        interpolated = cls.interpolate_11_point(pr_points)
        r_prec = cls.calculate_r_precision(retrieved, relevant)
        auc = cls.compute_auc(pr_points)

        return PRCurveData(
            query_id=query_id,
            pr_points=pr_points,
            interpolated_11_point=interpolated,
            r_precision=r_prec,
            auc=auc
        )
