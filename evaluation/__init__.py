"""
Evaluation package for IR system assessment
"""

from evaluation.metrics import RetrievalEvaluator, GenerationEvaluator
from evaluation.advanced_metrics import GradedRelevanceEvaluator, RobustMetricsCalculator
from evaluation.precision_recall import PrecisionRecallAnalyzer, PRCurveData
from evaluation.statistical_tests import StatisticalTester
from evaluation.trec_format import TRECFormatHandler
from evaluation.visualization import EvaluationVisualizer
from evaluation.reporting import EvaluationReporter

__all__ = [
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "GradedRelevanceEvaluator",
    "RobustMetricsCalculator",
    "PrecisionRecallAnalyzer",
    "PRCurveData",
    "StatisticalTester",
    "TRECFormatHandler",
    "EvaluationVisualizer",
    "EvaluationReporter",
]
