#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:38:18 2026

@author: wormulon
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

def run_pairwise_stats(data_mat, design_mat, group_names, test_type,
                       metric_name, stats_df, comp_name=""):
    """
    Same as before, but safe for first execution when stats_df is empty or None
    """

    # Ensure stats_df is a DataFrame
    if stats_df is None or not isinstance(stats_df, pd.DataFrame):
        stat_headers = ["metric","comparison","group1","group2",
                        "mean1","mean2","n1","n2","sd1","sd2",
                        "test-type","p-value","q-value"]
        stats_df = pd.DataFrame(columns=stat_headers)

    for comp in design_mat:
        i, j = comp

        g1 = data_mat[:, i]
        g2 = data_mat[:, j]

        # remove NaNs
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        n1, n2 = len(g1), len(g2)
        mean1, mean2 = np.mean(g1), np.mean(g2)
        sd1, sd2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

        if test_type.lower() == "ttest":
            stat, pval = stats.ttest_ind(g1, g2, equal_var=False)
            test_label = "ttest_ind"

        elif test_type.lower() == "mannwhitney":
            stat, pval = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            test_label = "mannwhitneyu"

        else:
            raise ValueError("test_type must be 'ttest' or 'mannwhitney'")

        new_row = {
            "metric": metric_name,
            "comparison": comp_name or f"{group_names[i]} vs {group_names[j]}",
            "group1": group_names[i],
            "group2": group_names[j],
            "mean1": mean1,
            "mean2": mean2,
            "n1": n1,
            "n2": n2,
            "sd1": sd1,
            "sd2": sd2,
            "test-type": test_label,
            "p-value": pval,
            "q-value": np.nan
        }

        # append row safely
        stats_df = pd.concat([stats_df, pd.DataFrame([new_row])],
                             ignore_index=True)

    return stats_df

def apply_bh_fdr(stats_df, p_col="p-value", q_col="q-value", sig_col="significance"):
    """
    Apply Benjamini-Hochberg FDR correction to the dataframe and add significance stars.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame containing at least a p-value column.
    p_col : str
        Name of the p-value column (default: "p-value").
    q_col : str
        Name of the q-value column to be filled (default: "q-value").
    sig_col : str
        Name of the column to indicate significance with stars (default: "significance").
        
    Returns
    -------
    stats_df : pd.DataFrame
        Updated DataFrame with q-values and significance stars.
    """

    pvals = stats_df[p_col].values
    n = len(pvals)

    # sort p-values
    order = np.argsort(pvals)
    ranked_pvals = pvals[order]

    # BH formula
    qvals = ranked_pvals * n / (np.arange(1, n + 1))

    # enforce monotonicity
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals[qvals > 1] = 1

    # put back into original order
    qvals_correct_order = np.empty_like(qvals)
    qvals_correct_order[order] = qvals

    stats_df[q_col] = qvals_correct_order

    # add significance stars
    def q_to_stars(q):
        if q < 0.001:
            return "***"
        elif q < 0.01:
            return "**"
        elif q < 0.05:
            return "*"
        else:
            return "ns"

    stats_df[sig_col] = stats_df[q_col].apply(q_to_stars)

    return stats_df


def two_samp_proportionality(mat):
    """
    Perform a two-sample test for proportions.
    
    Parameters
    ----------
    mat : np.ndarray, shape (2,2)
        mat[0,0] = number of successes in group 1
        mat[1,0] = total trials in group 1
        mat[0,1] = number of successes in group 2
        mat[1,1] = total trials in group 2

    Returns
    -------
    z : float
        z-statistic
    p : float
        two-tailed p-value
    """

    x1 = mat[0,0]
    n1 = mat[1,0]

    x2 = mat[0,1]
    n2 = mat[1,1]

    # sample proportions
    p1 = x1 / n1
    p2 = x2 / n2

    # pooled proportion
    p_hat = (x1 + x2) / (n1 + n2)

    # z-statistic
    z = (p1 - p2) / np.sqrt(p_hat * (1 - p_hat) * (1/n1 + 1/n2))

    # two-tailed p-value
    p = 2 * (1 - norm.cdf(abs(z)))

    return z, p
