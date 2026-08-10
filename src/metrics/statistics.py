import torch
import numpy as np
from typing import Dict, Union, Tuple, List
from scipy.stats import spearmanr, pearsonr, shapiro, wilcoxon, ttest_rel

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def compute_spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes Spearman's rank correlation coefficient (rho) and p-value."""
    res = spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def compute_pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes Pearson's correlation coefficient (r) and p-value."""
    res = pearsonr(x, y)
    return float(res.statistic), float(res.pvalue)


def check_normality(data: np.ndarray) -> Tuple[float, float, bool]:
    """
    Shapiro-Wilk test for normality.
    Returns (statistic, p-value, is_normal)
    is_normal is True if p > 0.05.
    """
    stat, p_val = shapiro(data)
    return float(stat), float(p_val), bool(p_val > 0.05)


def paired_statistical_test(
    condition_a: np.ndarray,
    condition_b: np.ndarray
) -> Dict[str, float]:
    """
    Performs paired statistical testing between two conditions on the same patient cohort:
    - Automatically checks normality (Shapiro-Wilk).
    - Runs Paired t-test if normal, or Wilcoxon signed-rank test if non-normal.
    - Computes Cohen's d effect size.
    """
    diff = condition_a - condition_b
    _, p_norm, is_normal = check_normality(diff)

    if is_normal:
        stat, p_val = ttest_rel(condition_a, condition_b)
        test_type = "Paired t-test"
    else:
        stat, p_val = wilcoxon(condition_a, condition_b)
        test_type = "Wilcoxon signed-rank"

    # Cohen's d effect size
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff != 0 else 0.0

    return {
        "p_value": float(p_val),
        "test_statistic": float(stat),
        "test_type": test_type,
        "is_normal": is_normal,
        "cohens_d": float(cohens_d)
    }


def apply_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """Applies Bonferroni multiple comparison correction across p-values."""
    if HAS_STATSMODELS:
        rejected, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
        return rejected.tolist(), corrected_p.tolist()
    else:
        n = len(p_values)
        corrected_p = [min(1.0, p * n) for p in p_values]
        rejected = [cp < alpha for cp in corrected_p]
        return rejected, corrected_p



def tost_equivalence_test(
    oracle_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    margin: float = 0.01
) -> Dict[str, float]:
    """
    Two One-Sided Tests (TOST) for equivalence (RQ1).
    Tests whether the difference (Oracle - Synthetic) is statistically bounded within [-margin, +margin].
    Default margin = 0.01 (1% Dice equivalence bound).
    """
    diff = oracle_scores - synthetic_scores
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se = std_diff / np.sqrt(n)

    # Upper bound test: H0_2: mean_diff >= margin (alternative: 'less')
    p_upper = float(ttest_rel(diff, np.full_like(diff, margin), alternative='less').pvalue)

    # Lower bound test: H0_1: mean_diff <= -margin (alternative: 'greater')
    p_lower = float(ttest_rel(diff, np.full_like(diff, -margin), alternative='greater').pvalue)

    # TOST p-value is the maximum of the two one-sided p-values
    p_tost = max(p_upper, p_lower)

    return {
        "p_tost": p_tost,
        "mean_difference": float(mean_diff),
        "margin": margin,
        "is_equivalent": bool(p_tost < 0.05)
    }
