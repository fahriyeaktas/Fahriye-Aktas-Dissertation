"""
robustness_checks.py
================================================================
Diagnostics, leave-one-out sensitivity analysis, and re-estimation of
the aggregate/decomposed models under two alternative textual measures:

  - FinBERT-FOMC sentiment (an alternative sentiment classifier)
  - Hawkish/dovish stance (a distinct communication construct, not a
    sentiment replacement)

No observation is mechanically deleted on the basis of Cook's distance;
influential observations are instead assessed via leave-one-out.
================================================================
"""
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
import statsmodels.formula.api as smf

from data_cleaning import ASSETS
from regressions import estimate_aggregate_model, estimate_decomposed_model, POLICY_MISSING_MEETINGS

POLICY_CONTROLS = ["MPC_Target", "MPC_Path", "MPC_QE"]


# ================================================================
# Diagnostics: correlation matrix, VIF, Cook's distance
# ================================================================

def compute_diagnostics(df, regressor_cols):
    correlation_matrix = df[regressor_cols].corr()
    X = df[regressor_cols].copy()
    X.insert(0, "const", 1.0)
    vif = {col: variance_inflation_factor(X.values, i + 1) for i, col in enumerate(regressor_cols)}
    return correlation_matrix, vif


def cooks_distance(model_formula, df):
    """Cook's distance and leverage for identification only; no
    observation is removed based on this diagnostic."""
    ols_fit = smf.ols(model_formula, data=df).fit()
    influence = OLSInfluence(ols_fit)
    return influence.cooks_distance[0], influence.hat_matrix_diag


# ================================================================
# Leave-one-out sensitivity analysis
# ================================================================

def leave_one_out_sensitivity(df, outcome_col, controls=None):
    """Re-estimate the controlled decomposed model once per meeting,
    omitting one meeting at a time. Descriptive sensitivity check;
    no observation is permanently removed from the main sample."""
    all_meeting_ids = df["Meeting_ID"].unique()
    results = []
    for omitted_id in all_meeting_ids:
        sub_df = df[df["Meeting_ID"] != omitted_id]
        model, sent_eq, unc_eq, joint_qa, joint_pr = estimate_decomposed_model(
            sub_df, outcome_col, controls=controls
        )
        results.append({
            "Omitted_Meeting_ID": omitted_id,
            "PR_Sentiment_beta": model.params["PR_Sentiment"],
            "PR_Sentiment_p": model.pvalues["PR_Sentiment"],
            "QA_Sentiment_beta": model.params["QA_Sentiment"],
            "QA_Sentiment_p": model.pvalues["QA_Sentiment"],
            "PR_QA_Sentiment_Equality_p": float(sent_eq.pvalue),
            "Joint_QA_p": float(joint_qa.pvalue),
            "Joint_PR_p": float(joint_pr.pvalue),
        })
    return pd.DataFrame(results)


# ================================================================
# Alternative textual-measure robustness
# Market outcomes, timing, uncertainty, controls, and estimator are
# unchanged; only the communication-content regressor is substituted.
# ================================================================

def run_alternative_measure_robustness(df, outcome_col, sentiment_col_map):
    """sentiment_col_map renames the alternative measure's columns to
    the Sentiment/PR_Sentiment/QA_Sentiment names expected by
    estimate_aggregate_model / estimate_decomposed_model, e.g.:
        {"Overall_Sentiment_FinBERT_FOMC": "Sentiment",
         "PR_Sentiment_FinBERT_FOMC": "PR_Sentiment",
         "QA_Sentiment_FinBERT_FOMC": "QA_Sentiment"}
    """
    renamed = df.rename(columns=sentiment_col_map)

    aggregate_model, aggregate_joint_wald = estimate_aggregate_model(renamed, outcome_col)
    aggregate_model_controlled, aggregate_joint_wald_controlled = estimate_aggregate_model(
        renamed, outcome_col, controls=POLICY_CONTROLS
    )

    decomposed_model, sent_eq, unc_eq, joint_qa, joint_pr = estimate_decomposed_model(renamed, outcome_col)
    (decomposed_model_controlled, sent_eq_controlled, unc_eq_controlled,
     joint_qa_controlled, joint_pr_controlled) = estimate_decomposed_model(
        renamed, outcome_col, controls=POLICY_CONTROLS
    )

    return {
        "aggregate": (aggregate_model, aggregate_joint_wald),
        "aggregate_controlled": (aggregate_model_controlled, aggregate_joint_wald_controlled),
        "decomposed": (decomposed_model, sent_eq, unc_eq, joint_qa, joint_pr),
        "decomposed_controlled": (
            decomposed_model_controlled, sent_eq_controlled, unc_eq_controlled,
            joint_qa_controlled, joint_pr_controlled,
        ),
    }


if __name__ == "__main__":
    # Example usage (requires whole_conference_df merged with text measures
    # from nlp_analysis.py and regressions.py):
    #
    # fomc_results = run_alternative_measure_robustness(
    #     robustness_df_fomc, f"{asset}_Overall_Return",
    #     {"Overall_Sentiment_FinBERT_FOMC": "Sentiment",
    #      "PR_Sentiment_FinBERT_FOMC": "PR_Sentiment",
    #      "QA_Sentiment_FinBERT_FOMC": "QA_Sentiment"},
    # )
    # hd_results = run_alternative_measure_robustness(
    #     robustness_df_hd, f"{asset}_Overall_Return",
    #     {"Overall_HD_Sentiment": "Sentiment",
    #      "PR_HD_Sentiment": "PR_Sentiment",
    #      "QA_HD_Sentiment": "QA_Sentiment"},
    # )
    pass
