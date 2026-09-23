"""
figures.py
================================================================
All matplotlib figure-generation functions used in the dissertation
and its supplementary appendix (Appendix E):

  - Paired Prepared Remarks vs Q&A scatter (45-degree reference line)
  - Distribution of a repricing measure by section (boxplot)
  - Paired difference with 95% bootstrap CI
  - Raw bivariate relationship (sentiment/uncertainty vs market outcome)
  - Coefficient comparison, PR vs Q&A (uncontrolled, single model)
  - Coefficient comparison, uncontrolled vs controlled (D1 vs D2)
  - Leave-one-out coefficient and p-value stability plots

Colour scheme is fixed throughout: PR/primary series in navy (#1F3864),
QA/secondary or controlled series in gold (#C1892F).
================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_cleaning import ASSETS

NAVY = "#1F3864"
GOLD = "#C1892F"

ASSET_UNIT = {
    "GBPUSD": "bp (log return)", "FTSE": "bp (log return)",
    "UK2Y": "bp (yield change)", "UK10Y": "bp (yield change)",
}


# ================================================================
# Prepared Remarks vs Q&A repricing figures
# ================================================================

def plot_paired_scatter(pr, qa, asset, measure_label, out_dir, stub):
    """45-degree reference line; points above the line indicate Q&A
    larger than Prepared Remarks for that meeting."""
    unit = ASSET_UNIT[asset]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pr, qa, alpha=0.7, color=NAVY, edgecolor="white", s=60)
    mx = max(pr.max(), qa.max()) * 1.05
    ax.plot([0, mx], [0, mx], linestyle="--", color="gray", linewidth=1, label="45-degree line (PR = QA)")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel(f"PR {measure_label} ({unit})", fontsize=10)
    ax.set_ylabel(f"QA {measure_label} ({unit})", fontsize=10)
    ax.set_title(f"Paired PR vs QA {measure_label} - {asset}\n(N={len(pr)})", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stub}_paired_{asset}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{stub}_paired_{asset}.pdf", bbox_inches="tight")
    plt.close()


def plot_distribution_boxplot(pr, qa, asset, measure_label, out_dir, stub):
    unit = ASSET_UNIT[asset]
    fig, ax = plt.subplots(figsize=(5, 6))
    bp = ax.boxplot([pr, qa], tick_labels=["PR", "QA"], patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [NAVY, GOLD]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(f"{measure_label} ({unit})", fontsize=10)
    ax.set_title(f"Distribution of {measure_label} - {asset}\n(N={len(pr)})", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stub}_distribution_{asset}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{stub}_distribution_{asset}.pdf", bbox_inches="tight")
    plt.close()


def plot_paired_difference_ci(mean_diff, boot_lo, boot_hi, asset, measure_label, out_dir, stub):
    unit = ASSET_UNIT[asset]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar([0], [mean_diff], yerr=[[mean_diff - boot_lo], [boot_hi - mean_diff]],
                fmt="o", color=NAVY, capsize=6, markersize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel(f"QA minus PR difference ({unit})", fontsize=10)
    ax.set_title(f"Paired Difference (QA-PR) with 95% Bootstrap CI - {asset}\n({measure_label})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stub}_diffCI_{asset}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{stub}_diffCI_{asset}.pdf", bbox_inches="tight")
    plt.close()


def make_prepared_remarks_qa_figures(out_dir, series_by_asset, results_df, measure_label, stub):
    """series_by_asset: {asset: (pr_series, qa_series)}.
    results_df: paired_comparison() output per asset, with an 'Asset'
    column, Mean_Diff, boot_ci_low, boot_ci_high."""
    for asset in ASSETS:
        pr, qa = series_by_asset[asset]
        plot_paired_scatter(pr, qa, asset, measure_label, out_dir, stub)
        plot_distribution_boxplot(pr, qa, asset, measure_label, out_dir, stub)
        row = results_df[results_df["Asset"] == asset].iloc[0]
        plot_paired_difference_ci(row["Mean_Diff"], row["boot_ci_low"], row["boot_ci_high"],
                                   asset, measure_label, out_dir, stub)


# ================================================================
# Raw bivariate relationship (aggregate model)
# ================================================================

def plot_raw_relationship(x, y, asset, xlabel, out_dir, fname_stub):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.7, color=NAVY, edgecolor="white", s=60)
    z = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, z[0] * xline + z[1], color=GOLD, linewidth=2, label="Raw fitted line (OLS, unadjusted)")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(f"{asset} Overall Return (signed, bp)", fontsize=10)
    ax.set_title(f"RAW relationship: {xlabel} vs {asset} Overall Return\n(N={len(x)}, bivariate)", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"raw_{fname_stub}_{asset}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"raw_{fname_stub}_{asset}.pdf", bbox_inches="tight")
    plt.close()


# ================================================================
# Coefficient comparison figures (decomposed model)
# ================================================================

def plot_pr_qa_coefficient_comparison(full_results_df, asset, variable_pair, ylabel, title_stub, out_dir, fname_stub):
    """variable_pair e.g. ("PR_Sent", "QA_Sent"); full_results_df has
    columns Asset, Variable, Beta, ci_lo, ci_hi."""
    row_full = full_results_df[full_results_df["Asset"] == asset].set_index("Variable")
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, v in enumerate(variable_pair):
        beta, lo, hi = row_full.loc[v, "Beta"], row_full.loc[v, "ci_lo"], row_full.loc[v, "ci_hi"]
        ax.errorbar([i], [beta], yerr=[[beta - lo], [hi - beta]],
                    fmt="o", color=NAVY if i == 0 else GOLD, capsize=6, markersize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"PR {ylabel}", f"QA {ylabel}"])
    ax.set_ylabel("Coefficient (95% CI, HC3)", fontsize=10)
    ax.set_title(f"PR vs QA {ylabel} Coefficient - {asset}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"coef_{fname_stub}_PRvsQA_{asset}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"coef_{fname_stub}_PRvsQA_{asset}.pdf", bbox_inches="tight")
    plt.close()


def plot_uncontrolled_vs_controlled(uncontrolled_vals, controlled_vals_with_ci, labels, asset,
                                     measure_name, out_dir, fname_stub):
    """uncontrolled_vals: list of point estimates (no CI shown, single
    model estimate). controlled_vals_with_ci: list of (beta, ci_lo, ci_hi).
    labels: e.g. ["D1 PR", "D2 PR", "D1 QA", "D2 QA"] — alternating
    uncontrolled/controlled pairs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, label in enumerate(labels):
        if i % 2 == 0:
            beta = uncontrolled_vals[i // 2]
            ax.errorbar([i], [beta], fmt="o", color=NAVY, capsize=6, markersize=9)
        else:
            beta, lo, hi = controlled_vals_with_ci[i // 2]
            ax.errorbar([i], [beta], yerr=[[beta - lo], [hi - beta]], fmt="o", color=GOLD, capsize=6, markersize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{measure_name} coefficient", fontsize=10)
    ax.set_title(f"Uncontrolled vs Controlled - {measure_name} - {asset}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname_stub}_{asset}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{fname_stub}_{asset}.pdf", bbox_inches="tight")
    plt.close()


# ================================================================
# Leave-one-out sensitivity figures
# ================================================================

def plot_leave_one_out_coefficient(loo_df, col_beta, full_sample_beta, title, out_dir, fname):
    fig, ax = plt.subplots(figsize=(12, 5))
    d = loo_df.sort_values("Omitted_Meeting_ID")
    ax.scatter(range(len(d)), d[col_beta], color=NAVY, s=25)
    ax.axhline(full_sample_beta, color=GOLD, linewidth=1.5, label="Full-sample coefficient")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["Omitted_Meeting_ID"], rotation=90, fontsize=6)
    ax.set_ylabel("Leave-one-out coefficient", fontsize=10)
    ax.set_xlabel("Omitted meeting", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close()


def plot_leave_one_out_pvalue(loo_df, col_p, full_sample_p, title, out_dir, fname):
    fig, ax = plt.subplots(figsize=(12, 4))
    d = loo_df.sort_values("Omitted_Meeting_ID")
    ax.scatter(range(len(d)), d[col_p], color=NAVY, s=25)
    ax.axhline(0.05, color="red", linewidth=1.2, linestyle="--", label="p = 0.05")
    ax.axhline(full_sample_p, color=GOLD, linewidth=1.5, label="Full-sample p-value")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["Omitted_Meeting_ID"], rotation=90, fontsize=6)
    ax.set_ylabel("p-value", fontsize=10)
    ax.set_xlabel("Omitted meeting", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    OUTPUT_DIR = Path("outputs") / "figures"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figure output directory ready: {OUTPUT_DIR}")
