import json
import warnings
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
SHAP_DIR = Path("notebooks/eda_output/shap")
SHAP_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    best_key = (MODELS_DIR / "best_model.txt").read_text().strip()
    model = joblib.load(MODELS_DIR / f"{best_key}.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    X_test = joblib.load(PROCESSED_DIR / "X_test.pkl")

    print(f"Best model: {best_key}")
    print(f"X_test shape: {X_test.shape}, using first 1000 rows for SHAP")
    X_shap = X_test[:1000]

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)

    joblib.dump(shap_values, MODELS_DIR / "shap_values.pkl")
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")

    sv = shap_values.values if hasattr(shap_values, "values") else shap_values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    # Top 20 features by mean |SHAP|
    mean_abs = np.abs(sv).mean(axis=0)
    top20_idx = np.argsort(mean_abs)[-20:][::-1]
    top_features = [feature_names[i] for i in top20_idx]

    print("\nTop 10 features by mean |SHAP|:")
    for i, idx in enumerate(top20_idx[:10]):
        print(f"  {i+1}. {feature_names[idx]}: {mean_abs[idx]:.4f}")

    # --- Plot 1: Summary Bar ---
    print("\nGenerating SHAP plots...")
    fig, ax = plt.subplots(figsize=(10, 8))
    top_mean = mean_abs[top20_idx]
    ax.barh(range(20), top_mean[::-1], color="#2980B9")
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_title("Top 20 Most Important Features (SHAP)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    save(fig, SHAP_DIR / "shap_summary_bar.png")

    # --- Plot 2: Beeswarm ---
    fig = plt.figure(figsize=(12, 9))
    shap.summary_plot(sv, X_shap, feature_names=feature_names,
                      max_display=20, show=False, plot_type="dot")
    plt.title("SHAP Beeswarm — Feature Impact Distribution", fontsize=13)
    save(fig, SHAP_DIR / "shap_summary_dot.png")

    # --- Plot 3: High-risk waterfall ---
    probs = model.predict_proba(X_shap)[:, 1]
    high_risk_idx = int(np.argmax(probs))
    low_risk_idx  = int(np.argmin(probs))

    for label, idx, fname in [
        ("High Risk", high_risk_idx, "shap_waterfall_high_risk.png"),
        ("Low Risk",  low_risk_idx,  "shap_waterfall_low_risk.png"),
    ]:
        fig = plt.figure(figsize=(12, 8))
        shap.waterfall_plot(shap_values[idx], max_display=15, show=False)
        plt.title(f"{label} Customer — Feature Contribution Breakdown", fontsize=13)
        save(fig, SHAP_DIR / fname)

    # --- Plot 4: Dependence plot for EXT_SOURCE_2 ---
    ext2_idx = feature_names.index("EXT_SOURCE_2") if "EXT_SOURCE_2" in feature_names else 0
    ext3_idx = feature_names.index("EXT_SOURCE_3") if "EXT_SOURCE_3" in feature_names else 0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_shap[:, ext2_idx], sv[:, ext2_idx],
               c=X_shap[:, ext3_idx], cmap="RdYlGn", alpha=0.5, s=10)
    ax.set_xlabel("EXT_SOURCE_2 value")
    ax.set_ylabel("SHAP value for EXT_SOURCE_2")
    ax.set_title("SHAP Dependence Plot — EXT_SOURCE_2 (colored by EXT_SOURCE_3)", fontsize=13)
    plt.colorbar(ax.collections[0], ax=ax, label="EXT_SOURCE_3")
    save(fig, SHAP_DIR / "shap_dependence_ext2.png")

    # Save top features list
    with open(MODELS_DIR / "top_features.json", "w") as f:
        json.dump(top_features, f, indent=2)

    print("\nInterpretation guide (top 5 features):")
    interpretations = {
        "EXT_SOURCE_2": "Higher score = lower default risk. Key external credit bureau signal.",
        "EXT_SOURCE_3": "Higher score = lower default risk. Third credit bureau score.",
        "EXT_SOURCE_1": "Higher score = lower default risk. First credit bureau score.",
        "EXT_SOURCE_MEAN": "Average of all external scores. Strong predictor of creditworthiness.",
        "EXT_SOURCE_MIN": "Worst of the three external scores. Flags credit weakness.",
        "CREDIT_INCOME_RATIO": "Higher ratio = more debt relative to income = higher risk.",
        "AMT_CREDIT": "Larger loans carry more default risk.",
        "IS_UNEMPLOYED": "Unemployed applicants default at 2x the rate.",
        "AGE_YEARS": "Younger applicants tend to have higher default rates.",
        "PAYMENT_RATE": "Higher monthly payment rate relative to loan = higher risk.",
    }
    for feat in top_features[:5]:
        print(f"  {feat}: {interpretations.get(feat, 'Higher values increase/decrease risk.')}")

    print("\nExplainability complete.")


if __name__ == "__main__":
    main()
