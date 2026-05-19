import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
PLOT_DIR = Path("notebooks/eda_output")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_train = joblib.load(PROCESSED_DIR / "X_train.pkl")
    X_val   = joblib.load(PROCESSED_DIR / "X_val.pkl")
    X_test  = joblib.load(PROCESSED_DIR / "X_test.pkl")
    y_train = joblib.load(PROCESSED_DIR / "y_train.pkl")
    y_val   = joblib.load(PROCESSED_DIR / "y_val.pkl")
    y_test  = joblib.load(PROCESSED_DIR / "y_test.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def find_optimal_threshold(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f = f1_score(y, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh, probs


def evaluate(name, model, X, y, threshold):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    roc = roc_auc_score(y, probs)
    pr  = average_precision_score(y, probs)
    f1  = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f2   = (1 + 4) * prec * rec / (4 * prec + rec + 1e-9)
    return {
        "model": name,
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f2": round(f2, 4),
        "threshold": round(threshold, 3),
        "probs": probs,
    }


def plot_roc_curves(results):
    y_val = results[0]["y_val"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in results:
        fpr, tpr, _ = roc_curve(y_val, r["probs"])
        ax.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend()
    fig.savefig(PLOT_DIR / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(results):
    y_val = results[0]["y_val"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in results:
        prec, rec, _ = precision_recall_curve(y_val, r["probs"])
        ax.plot(rec, prec, label=f"{r['model']} (PR-AUC={r['pr_auc']:.4f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend()
    fig.savefig(PLOT_DIR / "pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(model, X, y, threshold, model_name):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Repaid", "Predicted: Default"])
    ax.set_yticklabels(["Actual: Repaid", "Actual: Default"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion Matrix — {model_name} (threshold={threshold:.2f})")
    plt.colorbar(im)
    fig.savefig(PLOT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names, model_name, top_n=25):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = np.abs(model.coef_[0])
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(range(top_n), imp[idx], color="#2980B9")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14)
    ax.set_xlabel("Importance")
    fig.savefig(PLOT_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data()

    # --- Model 1: Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_thresh, _ = find_optimal_threshold(lr, X_val, y_val)
    lr_res = evaluate("Logistic Regression", lr, X_val, y_val, lr_thresh)
    lr_res["y_val"] = y_val

    # --- Model 2: XGBoost ---
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=11, eval_metric="auc", random_state=42,
        early_stopping_rounds=50, verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_thresh, _ = find_optimal_threshold(xgb, X_val, y_val)
    xgb_res = evaluate("XGBoost", xgb, X_val, y_val, xgb_thresh)
    xgb_res["y_val"] = y_val

    # Learning curve from XGBoost
    evals = xgb.evals_result()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(evals["validation_0"]["auc"], label="Validation AUC", color="#E74C3C")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("AUC")
    ax.set_title("XGBoost Learning Curve")
    ax.legend()
    fig.savefig(PLOT_DIR / "learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Model 3: LightGBM ---
    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        min_child_samples=20, is_unbalance=True, random_state=42,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(period=-1)],
    )
    lgbm_thresh, _ = find_optimal_threshold(lgbm, X_val, y_val)
    lgbm_res = evaluate("LightGBM", lgbm, X_val, y_val, lgbm_thresh)
    lgbm_res["y_val"] = y_val

    # --- Comparison ---
    results = [lr_res, xgb_res, lgbm_res]
    print("\n{'Model':<25} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1@opt':>8} {'Threshold':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<25} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['f1']:>8.4f} {r['threshold']:>10.3f}")

    best = max(results, key=lambda r: r["pr_auc"])
    model_map = {"Logistic Regression": (lr, "logistic_regression"),
                 "XGBoost": (xgb, "xgboost_credit"),
                 "LightGBM": (lgbm, "lightgbm_credit")}

    print(f"\nBest model: {best['model']} (PR-AUC={best['pr_auc']:.4f})")

    # Final test evaluation
    best_model_obj, best_model_key = model_map[best["model"]]
    test_res = evaluate(best["model"], best_model_obj, X_test, y_test, best["threshold"])
    print(f"\nTest set — ROC-AUC: {test_res['roc_auc']}, PR-AUC: {test_res['pr_auc']}, F1: {test_res['f1']}")
    probs_test = best_model_obj.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= best["threshold"]).astype(int)
    print(classification_report(y_test, preds_test, target_names=["Repaid", "Default"]))

    # Save test probabilities + labels for threshold analysis dashboard
    joblib.dump({"probs": probs_test, "y_true": y_test}, MODELS_DIR / "test_preds.pkl")

    # Plots
    print("Saving plots...")
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_confusion_matrix(best_model_obj, X_test, y_test, best["threshold"], best["model"])
    plot_feature_importance(best_model_obj, feature_names, best["model"])

    # Save models
    joblib.dump(lr, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(xgb, MODELS_DIR / "xgboost_credit.pkl")
    joblib.dump(lgbm, MODELS_DIR / "lightgbm_credit.pkl")
    (MODELS_DIR / "best_model.txt").write_text(best_model_key)

    thresholds = {
        "logistic_regression": lr_thresh,
        "xgboost_credit": xgb_thresh,
        "lightgbm_credit": lgbm_thresh,
    }
    with open(MODELS_DIR / "thresholds.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in thresholds.items()}, f, indent=2)

    metrics = {}
    for r, key in [(lr_res, "logistic_regression"), (xgb_res, "xgboost_credit"), (lgbm_res, "lightgbm_credit")]:
        metrics[key] = {k: v for k, v in r.items() if k not in ("probs", "y_val")}
    metrics["best_model"] = best_model_key
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll models and artifacts saved.")


if __name__ == "__main__":
    main()
