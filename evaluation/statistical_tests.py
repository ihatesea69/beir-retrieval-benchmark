"""
Statistical Testing for IR Evaluation
Implements significance testing and confidence intervals for system comparison
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class StatisticalTester:
    """
    Statistical testing utilities for IR system comparison
    """
    
    @staticmethod
    def paired_t_test(scores_a: List[float], scores_b: List[float]) -> Dict[str, float]:
        """
        Perform paired t-test between two systems' scores
        
        Args:
            scores_a: Per-query scores for system A
            scores_b: Per-query scores for system B
            
        Returns:
            Dict containing t-statistic, p-value, and degrees of freedom
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score lists must have equal length")
            
        if len(scores_a) < 2:
            raise ValueError("Need at least 2 paired observations")
        
        # Calculate differences
        differences = np.array(scores_a) - np.array(scores_b)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'degrees_of_freedom': len(scores_a) - 1,
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences, ddof=1))
        }
    
    @staticmethod
    def compute_confidence_interval(
        scores: List[float], 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for mean score
        
        Args:
            scores: List of scores
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(scores) < 2:
            raise ValueError("Need at least 2 observations for confidence interval")
            
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        sem = stats.sem(scores_array)  # Standard error of mean
        
        # Calculate confidence interval
        h = sem * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
        
        return (float(mean - h), float(mean + h))
    
    @staticmethod
    def bonferroni_correction(p_values: List[float]) -> List[float]:
        """
        Apply Bonferroni correction for multiple comparisons
        
        Args:
            p_values: List of uncorrected p-values
            
        Returns:
            List of Bonferroni-corrected p-values
        """
        if not p_values:
            return []
            
        n_comparisons = len(p_values)
        corrected = [min(p * n_comparisons, 1.0) for p in p_values]
        
        return corrected
    
    @staticmethod
    def effect_size_cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
        """
        Calculate Cohen's d effect size between two groups
        
        Args:
            scores_a: Scores for group A
            scores_b: Scores for group B
            
        Returns:
            Cohen's d effect size
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            raise ValueError("Need at least 2 observations per group")
            
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        
        # Pooled standard deviation
        var_a = np.var(scores_a, ddof=1)
        var_b = np.var(scores_b, ddof=1)
        n_a = len(scores_a)
        n_b = len(scores_b)
        
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return 0.0
            
        cohens_d = (mean_a - mean_b) / pooled_std
        return float(cohens_d)
    
    @staticmethod
    def interpret_significance(p_value: float) -> str:
        """
        Interpret statistical significance level
        
        Args:
            p_value: P-value from statistical test
            
        Returns:
            String interpretation of significance
        """
        if p_value < 0.001:
            return "*** (p < 0.001)"
        elif p_value < 0.01:
            return "** (p < 0.01)"
        elif p_value < 0.05:
            return "* (p < 0.05)"
        else:
            return "ns (not significant)"
    
    @staticmethod
    def interpret_effect_size(cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size
        
        Args:
            cohens_d: Cohen's d value
            
        Returns:
            String interpretation of effect size
        """
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def compare_systems(
        system_scores: Dict[str, List[float]], 
        alpha: float = 0.05
    ) -> "pd.DataFrame":
        """
        Compare multiple systems with pairwise statistical tests
        
        Args:
            system_scores: Dict mapping system names to per-query scores
            alpha: Significance level for tests
            
        Returns:
            DataFrame with pairwise comparison results
        """
        import pandas as pd
        
        import pandas as pd

        systems = list(system_scores.keys())
        n_systems = len(systems)
        
        if n_systems < 2:
            raise ValueError("Need at least 2 systems for comparison")
        
        results = []
        p_values = []
        
        for i in range(n_systems):
            for j in range(i + 1, n_systems):
                system_a = systems[i]
                system_b = systems[j]
                
                scores_a = system_scores[system_a]
                scores_b = system_scores[system_b]
                
                # Perform paired t-test
                test_result = StatisticalTester.paired_t_test(scores_a, scores_b)
                effect_size = StatisticalTester.effect_size_cohens_d(scores_a, scores_b)
                
                p_values.append(test_result['p_value'])
                
                results.append({
                    'system_a': system_a,
                    'system_b': system_b,
                    'mean_a': np.mean(scores_a),
                    'mean_b': np.mean(scores_b),
                    'mean_diff': test_result['mean_difference'],
                    't_statistic': test_result['t_statistic'],
                    'p_value': test_result['p_value'],
                    'cohens_d': effect_size,
                    'effect_size_interpretation': StatisticalTester.interpret_effect_size(effect_size)
                })
        
        # Apply Bonferroni correction
        corrected_p_values = StatisticalTester.bonferroni_correction(p_values)
        
        # Add corrected p-values and significance interpretations
        for i, result in enumerate(results):
            result['p_value_corrected'] = corrected_p_values[i]
            result['significance'] = StatisticalTester.interpret_significance(result['p_value'])
            result['significance_corrected'] = StatisticalTester.interpret_significance(corrected_p_values[i])
        
        return pd.DataFrame(results)