import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/raw/application_train.csv")
OUT_DIR = Path("notebooks/eda_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "CNT_CHILDREN",
    "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_YEAR", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
]


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_class_imbalance(df):
    counts = df["TARGET"].value_counts().sort_index()
    pcts = counts / len(df) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(["Repaid (0)", "Defaulted (1)"], counts, color=["#27AE60", "#E74C3C"])
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_title(f"Loan Default Rate ({df['TARGET'].mean():.2%})", fontsize=16, fontweight="bold")
    ax.set_ylabel("Count")
    save(fig, "class_imbalance.png")


def plot_income_credit(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    titles = ["Annual Income", "Loan Amount", "Loan Annuity", "Goods Price"]
    for ax, col, title in zip(axes.flat, cols, titles):
        data = df[col].dropna()
        ax.hist(np.log1p(data), bins=60, alpha=0.7, color="#2980B9")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("log(value)")
    fig.suptitle("Income & Loan Amount Distributions (log scale)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "income_credit_distribution.png")


def plot_age_employment(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    age = -df["DAYS_BIRTH"] / 365
    for t, color, label in [(0, "#27AE60", "Repaid"), (1, "#E74C3C", "Defaulted")]:
        ax1.hist(age[df["TARGET"] == t], bins=40, alpha=0.6, color=color, label=label)
    ax1.set_title("Age Distribution by Default Status")
    ax1.set_xlabel("Age (years)")
    ax1.legend()

    employed = df[df["DAYS_EMPLOYED"] != 365243].copy()
    emp_years = -employed["DAYS_EMPLOYED"] / 365
    for t, color, label in [(0, "#27AE60", "Repaid"), (1, "#E74C3C", "Defaulted")]:
        mask = employed["TARGET"] == t
        ax2.hist(emp_years[mask], bins=40, alpha=0.6, color=color, label=label)
    ax2.set_title("Employment Years (Employed Only)")
    ax2.set_xlabel("Years employed")
    ax2.legend()
    save(fig, "age_employment.png")


def plot_ext_sources(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, col in zip(axes, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
        plot_data = [df[df["TARGET"] == t][col].dropna() for t in [0, 1]]
        ax.boxplot(plot_data, labels=["Repaid", "Defaulted"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#AED6F1"),
                   medianprops=dict(color="#E74C3C", linewidth=2))
        ax.set_title(col)
        ax.set_ylabel("Score")
    fig.suptitle("External Credit Scores by Default Status", fontsize=15, fontweight="bold")
    save(fig, "ext_sources_boxplot.png")


def plot_correlation_heatmap(df):
    num = df[NUMERIC_COLS + ["TARGET"]].copy()
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="coolwarm", annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.3)
    ax.set_title("Correlation Heatmap (Numeric Features vs TARGET)", fontsize=14, fontweight="bold")
    save(fig, "correlation_heatmap.png")


def plot_categorical_default_rates(df):
    cats = ["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_CONTRACT_TYPE",
            "CODE_GENDER", "FLAG_OWN_CAR"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flat
    for ax, col in zip(axes, cats):
        rates = df.groupby(col)["TARGET"].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(rates)), rates.values, color="#3498DB")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels(rates.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Default Rate by {col.replace('NAME_', '').replace('_TYPE', '')}")
        ax.set_ylabel("Default Rate")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=7)
    axes[5].axis("off")
    fig.suptitle("Default Rates by Categorical Features", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "categorical_default_rates.png")


def plot_credit_income_ratio(df):
    df2 = df.copy()
    df2["CREDIT_INCOME_RATIO"] = df2["AMT_CREDIT"] / (df2["AMT_INCOME_TOTAL"] + 1)
    df2["ratio_bin"] = pd.cut(df2["CREDIT_INCOME_RATIO"], bins=20)
    rates = df2.groupby("ratio_bin", observed=True)["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(rates)), rates.values, marker="o", color="#E74C3C", linewidth=2)
    ax.axhline(df["TARGET"].mean(), linestyle="--", color="#7F8C8D", label="Average default rate")
    ax.set_title("Default Rate by Credit/Income Ratio", fontsize=14, fontweight="bold")
    ax.set_xlabel("Credit/Income Ratio (binned)")
    ax.set_ylabel("Default Rate")
    ax.legend()
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([str(b) for b in rates.index], rotation=45, ha="right", fontsize=7)
    save(fig, "credit_income_ratio.png")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"Default rate: {df['TARGET'].mean():.2%}")

    high_null = df.isnull().mean()
    high_null_cols = high_null[high_null > 0.2].index.tolist()
    print(f"\nColumns with >20% missing: {high_null_cols}")

    num_df = df[NUMERIC_COLS].copy()
    corr_with_target = num_df.corrwith(df["TARGET"]).abs().sort_values(ascending=False)
    print("\nTop 10 features by correlation with TARGET:")
    print(corr_with_target.head(10).to_string())

    avg_loan = df.groupby("TARGET")["AMT_CREDIT"].mean()
    print(f"\nAvg loan — Non-default: {avg_loan[0]:,.0f}, Default: {avg_loan[1]:,.0f}")

    for col in ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"]:
        print(f"\nDefault rate by {col}:")
        print(df.groupby(col)["TARGET"].mean().sort_values(ascending=False).to_string())

    summary = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "default_rate": float(df["TARGET"].mean()),
        "avg_income": float(df["AMT_INCOME_TOTAL"].mean()),
        "avg_credit": float(df["AMT_CREDIT"].mean()),
        "avg_loan_nondefault": float(avg_loan[0]),
        "avg_loan_default": float(avg_loan[1]),
        "high_null_cols": high_null_cols,
        "top_10_corr_features": corr_with_target.head(10).to_dict(),
    }
    with open(OUT_DIR / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary_stats.json")

    print("\nGenerating plots...")
    plot_class_imbalance(df)
    plot_income_credit(df)
    plot_age_employment(df)
    plot_ext_sources(df)
    plot_correlation_heatmap(df)
    plot_categorical_default_rates(df)
    plot_credit_income_ratio(df)
    print("\nAll 7 plots saved.")


if __name__ == "__main__":
    main()
