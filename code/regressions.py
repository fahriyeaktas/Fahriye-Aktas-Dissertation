"""
regressions.py
================================================================
Constructs market outcomes (Prepared Remarks vs Q&A repricing measures
and the whole-conference regression outcome) and estimates the primary
models:

  - Prepared Remarks vs Q&A statistical comparison (paired t-test,
    Wilcoxon signed-rank test, paired bootstrap 95% CI)
  - Aggregate communication model
  - Decomposed Prepared Remarks-Q&A model, with the direct PR-vs-Q&A
    equality Wald test and section-level joint Wald tests
  - Policy-surprise-controlled variants of both models

All regressions use OLS with HC3 heteroskedasticity-robust standard
errors.
================================================================
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from data_cleaning import (
    ASSETS, SEC_PREFIX, LOG_RETURN_ASSETS, POLICY_SURPRISE_VARS,
    load_event_timing, load_market_data,
)

np.random.seed(42)

POLICY_MISSING_MEETINGS = ["2020_Q2", "2020_Q3"]  # excluded from the N=36 controlled sample


# ================================================================
# Market outcome construction
# ================================================================

MAX_MATCH_GAP_MINUTES = 2.0  # flag (do not silently accept) a match further than this from the target


def price_at_or_before(market_df, target_dt, max_gap_minutes=MAX_MATCH_GAP_MINUTES):
    """Nearest-minute matching rule: use the last available observation
    at or before the target timestamp. Never interpolates, zero-fills,
    or uses a future observation. With one-minute market data and
    nearest-minute event timing, the gap between the target and the
    matched observation should never exceed ~60 seconds; this was
    verified directly (max observed gap = 30 seconds) rather than
    assumed. max_gap_minutes is a safety check against market-data
    gaps, not a substitute for that verification."""
    sub = market_df[market_df["Timestamp"] <= target_dt]
    if len(sub) == 0:
        return None, None
    matched = sub.iloc[-1]
    gap_minutes = (target_dt - matched["Timestamp"]).total_seconds() / 60
    if gap_minutes > max_gap_minutes:
        print(f"WARNING: matched observation is {gap_minutes:.1f} min before target "
              f"{target_dt} (exceeds {max_gap_minutes} min tolerance) - check for a market-data gap.")
    return matched["Price"], gap_minutes


def signed_change(start_value, end_value, is_log_return):
    """GBP/USD and FTSE: 10000 x ln(end/start). Gilt yields:
    100 x (end - start), yields in percentage points."""
    if start_value is None or end_value is None:
        return None
    if is_log_return:
        return 10000 * np.log(end_value / start_value)
    return 100 * (end_value - start_value)


def one_minute_changes(price_series, is_log_return):
    p = price_series.values
    if len(p) < 2:
        return np.array([])
    return 10000 * np.log(p[1:] / p[:-1]) if is_log_return else 100 * (p[1:] - p[:-1])


def duration_adjusted_net_repricing(signed_return, duration_minutes):
    """Primary Prepared-Remarks-vs-Q&A measure: M = |signed return or
    change| / section duration (minutes). This is net repricing per
    minute; it is NOT realised volatility."""
    return abs(signed_return) / duration_minutes


def duration_normalised_realised_volatility(changes):
    """Secondary measure: RV_norm = sqrt(mean(one-minute changes^2))
    within the section window."""
    if len(changes) == 0:
        return None
    return np.sqrt(np.mean(changes ** 2))


def build_whole_conference_outcomes(timing_buffered, intraday):
    """Whole-conference regression outcome (RQ2-4 dependent variable):
    PR start - 5 min to Q&A end + 10 min, nearest-minute matched.
    SIGNED throughout (never absolute) - this is a deliberate, final
    specification choice: RQ2 explicitly tests directional communication
    effects, not repricing magnitude. (RQ1's duration-adjusted net
    repricing, built separately below, is the magnitude-based measure.)"""
    outcomes = []
    for _, row in timing_buffered.iterrows():
        mid = row["Meeting_ID"]
        window_start = row["Overall_Start_Rounded_Time"]
        window_end = row["Overall_End_Rounded_Time"]
        result = {"Meeting_ID": mid}
        for market_key, asset_prefix in SEC_PREFIX.items():
            market_series = intraday[market_key].get(mid)
            is_log = market_key in LOG_RETURN_ASSETS
            start_value, _ = price_at_or_before(market_series, window_start) if market_series is not None else (None, None)
            end_value, _ = price_at_or_before(market_series, window_end) if market_series is not None else (None, None)
            result[f"{asset_prefix}_Overall_Return"] = signed_change(start_value, end_value, is_log)
        outcomes.append(result)
    return pd.DataFrame(outcomes)


def build_section_repricing_measures(timing_unbuffered, intraday):
    """Section-specific PR and Q&A measures, for each asset:
      - PRIMARY: duration-adjusted net repricing, |signed change| / duration
      - SECONDARY: duration-normalised realised volatility
    Both are computed here directly from matched section-boundary prices
    and the intraminute price sequence, so this function does not depend
    on any pre-existing PR/QA return columns elsewhere in the workbook."""
    rows = []
    for _, row in timing_unbuffered.iterrows():
        mid = row["Meeting_ID"]
        windows = {
            "PR": (row["PR_Start_Rounded_Time"], row["PR_End_Rounded_Time"], row["PR_Duration_Min"]),
            "QA": (row["QA_Start_Rounded_Time"], row["QA_End_Rounded_Time"], row["QA_Duration_Min"]),
        }
        for market_key, asset_prefix in SEC_PREFIX.items():
            market_series = intraday[market_key].get(mid)
            is_log = market_key in LOG_RETURN_ASSETS
            result = {"Meeting_ID": mid, "Asset": asset_prefix}
            for section, (w_start, w_end, duration_min) in windows.items():
                if market_series is not None:
                    start_value, _ = price_at_or_before(market_series, w_start)
                    end_value, _ = price_at_or_before(market_series, w_end)
                    section_change = signed_change(start_value, end_value, is_log)
                    sub = market_series[(market_series["Timestamp"] >= w_start) & (market_series["Timestamp"] <= w_end)]
                    changes = one_minute_changes(sub["Price"], is_log)
                else:
                    section_change, changes = None, np.array([])

                result[f"{section}_NetRepricing"] = (
                    duration_adjusted_net_repricing(section_change, duration_min)
                    if section_change is not None else None
                )
                result[f"{section}_RV"] = duration_normalised_realised_volatility(changes)
            rows.append(result)
    return pd.DataFrame(rows)


# ================================================================
# Prepared Remarks vs Q&A statistical comparison
# ================================================================

def bootstrap_paired_ci(differences, n_replications=10000, alpha=0.05):
    differences = np.asarray(differences)
    n = len(differences)
    boot_means = np.array([
        np.random.choice(differences, n, replace=True).mean() for _ in range(n_replications)
    ])
    return np.percentile(boot_means, 100 * alpha / 2), np.percentile(boot_means, 100 * (1 - alpha / 2))


def paired_comparison(pr_values, qa_values):
    """Q&A minus Prepared Remarks. Negative values indicate greater
    repricing during Prepared Remarks. Explicitly aligns on a common,
    complete-case index first, since a paired test requires every
    observation to have both a PR and a QA value."""
    paired = pd.DataFrame({"PR": pr_values, "QA": qa_values}).dropna()
    pr_values, qa_values = paired["PR"], paired["QA"]
    diff = qa_values - pr_values
    t_stat, t_p = stats.ttest_rel(qa_values, pr_values)
    w_stat, w_p = stats.wilcoxon(qa_values, pr_values)
    ci_low, ci_high = bootstrap_paired_ci(diff.values)
    return {
        "N": len(diff), "PR_mean": pr_values.mean(), "QA_mean": qa_values.mean(),
        "Mean_Diff": diff.mean(), "t_stat": t_stat, "t_p": t_p, "w_stat": w_stat, "w_p": w_p,
        "boot_ci_low": ci_low, "boot_ci_high": ci_high,
        "N_PR_larger": int((diff < 0).sum()), "N_QA_larger": int((diff > 0).sum()),
    }


# ================================================================
# Aggregate and decomposed regression models (OLS, HC3)
# ================================================================

def estimate_aggregate_model(df, outcome_col, controls=None):
    formula_vars = ["Sentiment", "Uncertainty"] + (controls or [])
    sub = df[[outcome_col] + formula_vars].rename(columns={outcome_col: "Y"}).dropna()
    formula = "Y ~ " + " + ".join(formula_vars)
    model = smf.ols(formula, data=sub).fit(cov_type="HC3")
    joint_wald = model.wald_test("Sentiment = 0, Uncertainty = 0", scalar=True)
    return model, joint_wald


def estimate_decomposed_model(df, outcome_col, controls=None):
    formula_vars = ["PR_Sentiment", "QA_Sentiment", "PR_Uncertainty", "QA_Uncertainty"] + (controls or [])
    sub = df[[outcome_col] + formula_vars].rename(columns={outcome_col: "Y"}).dropna()
    formula = "Y ~ " + " + ".join(formula_vars)
    model = smf.ols(formula, data=sub).fit(cov_type="HC3")

    # Direct PR-vs-Q&A coefficient-equality Wald tests
    sentiment_equality = model.t_test("PR_Sentiment = QA_Sentiment")
    uncertainty_equality = model.t_test("PR_Uncertainty = QA_Uncertainty")

    # Section-level joint Wald tests
    joint_qa = model.wald_test("QA_Sentiment = 0, QA_Uncertainty = 0", scalar=True)
    joint_pr = model.wald_test("PR_Sentiment = 0, PR_Uncertainty = 0", scalar=True)

    return model, sentiment_equality, uncertainty_equality, joint_qa, joint_pr


# ================================================================
# Final dataset construction: merge whole-conference outcomes with text
# measures and policy controls, and rename to the generic column names
# (Sentiment, Uncertainty, PR_Sentiment, ...) expected by the model
# functions above. Every merge is one-to-one on Meeting_ID and the
# resulting N is checked explicitly rather than assumed.
# ================================================================

TEXT_MEASURE_RENAME = {
    "Overall_Sentiment_RoBERTa": "Sentiment", "Overall_Uncertainty_LM": "Uncertainty",
    "PR_Sentiment_RoBERTa": "PR_Sentiment", "QA_Sentiment_RoBERTa": "QA_Sentiment",
    "PR_Uncertainty_LM": "PR_Uncertainty", "QA_Uncertainty_LM": "QA_Uncertainty",
}


def build_analysis_dataset(whole_conference_df, roberta_df, uncertainty_df, policy_df):
    """One-to-one merge on Meeting_ID; renames to the generic regressor
    names; returns both the uncontrolled dataset (expected N=38) and the
    policy-controlled dataset (expected N=36, complete cases only)."""
    n_start = len(whole_conference_df)

    merged = whole_conference_df.merge(roberta_df, on="Meeting_ID", how="left", validate="one_to_one")
    assert len(merged) == n_start, "Merge with roberta_df changed row count - check for duplicate Meeting_ID"

    merged = merged.merge(uncertainty_df, on="Meeting_ID", how="left", validate="one_to_one")
    assert len(merged) == n_start, "Merge with uncertainty_df changed row count - check for duplicate Meeting_ID"

    merged = merged.rename(columns=TEXT_MEASURE_RENAME)

    uncontrolled_df = merged.copy()
    print(f"Uncontrolled dataset: N = {len(uncontrolled_df)} (expected 38)")
    assert len(uncontrolled_df) == 38, f"Expected N=38 for the uncontrolled dataset, got {len(uncontrolled_df)}"

    controlled_df = merged.merge(policy_df, on="Meeting_ID", how="left", validate="one_to_one")
    controlled_df = controlled_df.dropna(subset=POLICY_SURPRISE_VARS)
    print(f"Controlled dataset: N = {len(controlled_df)} (expected 36, after dropping meetings "
          f"missing policy-surprise data: {POLICY_MISSING_MEETINGS})")
    assert len(controlled_df) == 36, f"Expected N=36 for the controlled dataset, got {len(controlled_df)}"

    return uncontrolled_df, controlled_df


if __name__ == "__main__":
    from nlp_analysis import compute_roberta_sentiment, compute_lm_uncertainty
    from data_cleaning import load_transcript_files, load_policy_surprises

    meeting_master, timing_unbuffered, timing_buffered = load_event_timing()
    intraday = load_market_data()
    pr_files, qa_files, meeting_ids = load_transcript_files()
    policy_df = load_policy_surprises()

    whole_conference_df = build_whole_conference_outcomes(timing_buffered, intraday)
    section_repricing_df = build_section_repricing_measures(timing_unbuffered, intraday)
    roberta_df = compute_roberta_sentiment(pr_files, qa_files, meeting_ids)
    uncertainty_df = compute_lm_uncertainty(pr_files, qa_files, meeting_ids)

    uncontrolled_df, controlled_df = build_analysis_dataset(whole_conference_df, roberta_df, uncertainty_df, policy_df)

    print(f"Whole-conference outcomes: {whole_conference_df.shape}")
    print(f"Section repricing measures: {section_repricing_df.shape}")
