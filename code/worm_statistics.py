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
    Apply Benjamini-Hochberg FDR correction to the dataframe
    while safely ignoring NaN p-values.
    """

    pvals = stats_df[p_col].to_numpy(dtype=float)

    # --- Step 1: identify valid p-values
    valid_mask = ~np.isnan(pvals)
    valid_pvals = pvals[valid_mask]

    # If nothing to correct, just fill columns and exit
    if valid_pvals.size == 0:
        stats_df[q_col] = np.nan
        stats_df[sig_col] = "ns"
        return stats_df

    # --- Step 2: do BH ONLY on valid p-values
    n = valid_pvals.size
    order = np.argsort(valid_pvals)
    ranked_pvals = valid_pvals[order]

    qvals = ranked_pvals * n / (np.arange(1, n + 1))
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals[qvals > 1] = 1

    # restore original order within valid set
    qvals_correct = np.empty_like(qvals)
    qvals_correct[order] = qvals

    # --- Step 3: place q-values back into full array
    full_qvals = np.full_like(pvals, np.nan)
    full_qvals[valid_mask] = qvals_correct

    stats_df[q_col] = full_qvals

    # --- Step 4: add stars
    def q_to_stars(q):
        if np.isnan(q):
            return "ns"
        elif q < 0.001:
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

def proportionality_stats_from_df(df, design_mat, stats_df,
                                  comparison_name="overall"):
    """
    Perform pairwise proportionality tests for each larval stage
    between groups defined in design_mat.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe as described.
    design_mat : list of [i, j]
        Indices of groups to compare (row indices of df).
    stats_df : pd.DataFrame
        Empty or existing dataframe to append results to.
    comparison_name : str
        Label for the 'comparison' column (default: "overall")

    Returns
    -------
    stats_df : pd.DataFrame
        Updated dataframe with appended results.
    """

    # All stage columns (exclude 'group' and 'sum')
    stage_cols = [c for c in df.columns if c not in ["group", "sum"]]

    for stage in stage_cols:
        for comp in design_mat:
            i, j = comp

            group1 = df.loc[i, "group"]
            group2 = df.loc[j, "group"]

            x1 = df.loc[i, stage]
            n1 = df.loc[i, "sum"]

            x2 = df.loc[j, stage]
            n2 = df.loc[j, "sum"]

            mat = np.array([[x1, x2],
                            [n1, n2]])

            z, p = two_samp_proportionality(mat)

            new_row = {
                "Stage": stage,
                "comparison": comparison_name,
                "group1": group1,
                "group2": group2,
                "x1": x1,
                "x2": x2,
                "n1": n1,
                "n2": n2,
                "p-value": p,
                "q-value": np.nan
            }

            stats_df = pd.concat([stats_df, pd.DataFrame([new_row])],
                                 ignore_index=True)

    return stats_df