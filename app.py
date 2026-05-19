import base64
import io
import json
import os
import re
import warnings
from functools import wraps
from pathlib import Path

import bcrypt
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from flask import (Flask, jsonify, redirect, render_template,
                   request, session, url_for)

load_dotenv()
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-fallback-key")

BASE       = Path(__file__).parent
MODELS_DIR = BASE / "models"
PLOT_DIR   = BASE / "notebooks" / "eda_output"
SHAP_DIR   = PLOT_DIR / "shap"
USERS_FILE = BASE / "users.json"


# ── user store ────────────────────────────────────────────────────────────────

def load_users() -> dict:
    with open(USERS_FILE) as f:
        return {u["username"]: u for u in json.load(f)["users"]}

def save_user(username: str, full_name: str, password: str, role: str = "viewer"):
    with open(USERS_FILE) as f:
        data = json.load(f)
    data["users"].append({
        "username": username,
        "full_name": full_name,
        "password_hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
        "role": role,
    })
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── ML artifacts (lazy) ───────────────────────────────────────────────────────

_art = {}

def get_artifacts():
    if _art:
        return _art
    best_key = (MODELS_DIR / "best_model.txt").read_text().strip()
    _art["model"]      = joblib.load(MODELS_DIR / f"{best_key}.pkl")
    _art["explainer"]  = joblib.load(MODELS_DIR / "shap_explainer.pkl")
    _art["prep"]       = joblib.load(MODELS_DIR / "preprocessor.pkl")
    _art["feat_names"] = joblib.load(MODELS_DIR / "feature_names.pkl")
    _art["best_key"]   = best_key
    with open(MODELS_DIR / "thresholds.json") as f:
        _art["threshold"] = json.load(f).get(best_key, 0.5)
    return _art

def img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── feature engineering ───────────────────────────────────────────────────────

def build_input_df(form: dict) -> pd.DataFrame:
    age         = max(18, min(80, int(form["age"])))
    income      = max(10000, float(form["income"]))
    loan_amt    = max(10000, float(form["loan_amt"]))
    n_children  = int(form.get("n_children", 0))
    years_emp   = float(form.get("years_employed", 0))
    emp_type    = form["emp_type"]
    gender      = form["gender"]
    education   = form["education"]
    fam_status  = form["family_status"]
    own_car     = str(form.get("own_car", "0")) == "1"
    own_realty  = str(form.get("own_realty", "0")) == "1"
    n_inq       = float(form.get("n_inquiries", 2))

    # Bureau history features (0 = no credit history on file)
    bureau_cnt           = float(form.get("bureau_cnt", 0))
    bureau_cnt_active    = float(form.get("bureau_cnt_active", 0))
    bureau_overdue_max   = float(form.get("bureau_overdue_max", 0))
    bureau_amt_debt      = float(form.get("bureau_amt_debt", 0))
    bureau_amt_overdue   = float(form.get("bureau_amt_overdue_sum", 0))
    bureau_max_ever      = float(form.get("bureau_max_overdue_ever", 0))
    bureau_prolong       = float(form.get("bureau_prolong_sum", 0))
    bureau_days_mean     = float(form.get("bureau_days_credit_mean", -500))

    # CIBIL 300-900 → normalized 0-1; fall back to direct ext2 if already normalized
    raw_ext2 = float(form.get("ext2", 0.5))
    ext2 = (raw_ext2 - 300) / 600 if raw_ext2 > 1 else raw_ext2
    ext1 = float(form.get("ext1", 0.5))
    ext3 = float(form.get("ext3", 0.5))
    # Keep ext1/ext3 in 0-1; if user sent 0-100 scale, normalize
    if ext1 > 1: ext1 = ext1 / 100
    if ext3 > 1: ext3 = ext3 / 100
    ext1 = max(0.0, min(1.0, ext1))
    ext2 = max(0.0, min(1.0, ext2))
    ext3 = max(0.0, min(1.0, ext3))

    # Annuity: enforce minimum so CREDIT_TERM and PAYMENT_RATE stay sane
    raw_annuity = float(form.get("annuity", 0))
    annuity = max(raw_annuity, loan_amt * 0.015)   # at least 1.5% of loan per month

    days_employed = 365243 if emp_type == "Unemployed" else -int(years_emp * 365)
    income_type   = {"Employed": "Working", "Self-employed": "Commercial associate",
                     "Unemployed": "Working", "Pensioner": "Pensioner"}.get(emp_type, "Working")

    # Engineered features — clipped to training distribution to prevent OOD extrapolation
    credit_income_ratio = np.clip(loan_amt / (income + 1),         0.05, 25.0)
    annuity_income_ratio= np.clip(annuity  / (income + 1),         0.001, 0.5)
    credit_term         = np.clip(loan_amt / (annuity + 1),        6.0, 360.0)
    payment_rate        = np.clip(annuity  / (loan_amt + 1),       0.005, 0.20)
    income_per_child    = income / (n_children + 1)

    row = {
        "NAME_CONTRACT_TYPE":          "Cash loans",
        "CODE_GENDER":                 "M" if gender == "Male" else "F",
        "FLAG_OWN_CAR":                "Y" if own_car  else "N",
        "FLAG_OWN_REALTY":             "Y" if own_realty else "N",
        "CNT_CHILDREN":                n_children,
        "AMT_INCOME_TOTAL":            income,
        "AMT_CREDIT":                  loan_amt,
        "AMT_ANNUITY":                 annuity,
        "AMT_GOODS_PRICE":             loan_amt * 0.9,
        "NAME_EDUCATION_TYPE":         education,
        "NAME_FAMILY_STATUS":          fam_status,
        "NAME_INCOME_TYPE":            income_type,
        "NAME_HOUSING_TYPE":           "House / apartment",
        "DAYS_BIRTH":                  -age * 365,
        "DAYS_EMPLOYED":               days_employed,
        "DAYS_REGISTRATION":           -5000,
        "CNT_FAM_MEMBERS":             n_children + 2,
        "EXT_SOURCE_1":                ext1,
        "EXT_SOURCE_2":                ext2,
        "EXT_SOURCE_3":                ext3,
        "AMT_REQ_CREDIT_BUREAU_YEAR":  n_inq,
        "OBS_30_CNT_SOCIAL_CIRCLE":    2.0,
        "DEF_30_CNT_SOCIAL_CIRCLE":    0.0,
        "ORGANIZATION_TYPE":           "Business Entity Type 3",
        # Bureau aggregate features
        "BUREAU_CNT":               bureau_cnt,
        "BUREAU_CNT_ACTIVE":        bureau_cnt_active,
        "BUREAU_DAYS_CREDIT_MEAN":  bureau_days_mean,
        "BUREAU_OVERDUE_MAX":       bureau_overdue_max,
        "BUREAU_AMT_OVERDUE_SUM":   bureau_amt_overdue,
        "BUREAU_AMT_DEBT_SUM":      bureau_amt_debt,
        "BUREAU_MAX_OVERDUE_EVER":  bureau_max_ever,
        "BUREAU_PROLONG_SUM":       bureau_prolong,
        # bureau_balance payment history features
        "BB_MAX_DPD_MAX":      float(form.get("bb_max_dpd",       0)),
        "BB_MEAN_DPD_MEAN":    float(form.get("bb_max_dpd",       0)) * 0.3,
        "BB_LATE_COUNT_SUM":   float(form.get("bb_late_count",    0)),
        "BB_SEVERE_COUNT_SUM": float(form.get("bb_severe_count",  0)),
        "BB_LATE_RECENT_SUM":  float(form.get("bb_late_recent",   0)),
        "BB_MONTHS_TOTAL":     float(form.get("bb_months_total",  0)),
        "CREDIT_INCOME_RATIO":         credit_income_ratio,
        "ANNUITY_INCOME_RATIO":        annuity_income_ratio,
        "CREDIT_TERM":                 credit_term,
        "AGE_YEARS":                   float(age),
        "EMPLOYMENT_YEARS":            -1 if emp_type == "Unemployed" else years_emp,
        "IS_UNEMPLOYED":               1 if emp_type == "Unemployed" else 0,
        "EXT_SOURCE_MEAN":             float(np.nanmean([ext1, ext2, ext3])),
        "EXT_SOURCE_MIN":              float(np.nanmin([ext1, ext2, ext3])),
        "INCOME_PER_CHILD":            income_per_child,
        "PAYMENT_RATE":                payment_rate,
    }
    return pd.DataFrame([row])


def preprocess(raw_df, prep):
    num_cols = prep.named_transformers_["num"].feature_names_in_.tolist()
    cat_cols = prep.named_transformers_["cat"].feature_names_in_.tolist()
    for c in num_cols + cat_cols:
        if c not in raw_df.columns:
            raw_df[c] = 0
    return prep.transform(raw_df[num_cols + cat_cols])


def shap_waterfall_b64(explainer, X_arr, feat_names):
    sv = explainer(X_arr)
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv_row = shap.Explanation(
            values=sv.values[0, :, 1],
            base_values=sv.base_values[0, 1],
            data=sv.data[0],
            feature_names=feat_names,
        )
    else:
        sv_row = sv[0]
    fig = plt.figure(figsize=(9, 6))
    shap.waterfall_plot(sv_row, max_display=10, show=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def get_shap_factors(explainer, X_arr, feat_names, top_n=8):
    sv = explainer(X_arr)
    vals = sv.values[0, :, 1] if (hasattr(sv, "values") and sv.values.ndim == 3) else sv.values[0]
    idx  = np.argsort(np.abs(vals))[::-1][:top_n]
    return [{"feature": feat_names[i], "shap": round(float(vals[i]), 4),
              "value": round(float(X_arr[0, i]), 4)} for i in idx]


def risk_info(prob, threshold):
    # Threshold-relative risk zones: LOW = safe zone, MEDIUM = gray zone, HIGH = reject zone
    low_boundary = threshold * 0.60   # e.g. 0.77 * 0.60 ≈ 0.46
    if prob < low_boundary: label, cls = "LOW RISK",    "low"
    elif prob < threshold:  label, cls = "MEDIUM RISK", "medium"
    else:                   label, cls = "HIGH RISK",   "high"
    # Only HIGH RISK (above threshold) goes to REVIEW
    approved = cls != "high"
    return label, cls, approved


# ── auth routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("dashboard") if session.get("logged_in") else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = load_users()
        if username in users and bcrypt.checkpw(password.encode(),
                                                 users[username]["password_hash"].encode()):
            session.update(logged_in=True, username=username,
                           full_name=users[username]["full_name"],
                           role=users[username]["role"])
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html", error=None)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username  = request.form.get("username", "").strip().lower()
        password  = request.form.get("password", "")
        confirm   = request.form.get("confirm", "")

        if not re.match(r"^[a-z0-9_]{3,20}$", username):
            return render_template("signup.html",
                error="Username must be 3-20 chars, letters/numbers/underscores only.")
        if len(password) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters.")
        if password != confirm:
            return render_template("signup.html", error="Passwords do not match.")
        if username in load_users():
            return render_template("signup.html", error="Username already taken.")

        save_user(username, full_name, password)
        session.update(logged_in=True, username=username,
                       full_name=full_name, role="viewer")
        return redirect(url_for("dashboard"))
    return render_template("signup.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── page routes ───────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    with open(MODELS_DIR / "model_metrics.json") as f: metrics = json.load(f)
    with open(MODELS_DIR / "thresholds.json")    as f: thresholds = json.load(f)

    best_key = metrics.get("best_model", "xgboost_credit")
    best     = metrics.get(best_key, {})
    labels   = {"logistic_regression": "Logistic Regression",
                 "xgboost_credit": "XGBoost", "lightgbm_credit": "LightGBM"}

    table = []
    for k in ["logistic_regression", "xgboost_credit", "lightgbm_credit"]:
        m = metrics.get(k, {})
        table.append({"name": labels[k], "key": k, "best": k == best_key,
                      "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"),
                      "f1": m.get("f1"), "precision": m.get("precision"),
                      "recall": m.get("recall"), "threshold": thresholds.get(k)})

    plots = {}
    for n in ["roc_curves","pr_curves","confusion_matrix","feature_importance",
              "learning_curve","class_imbalance","ext_sources_boxplot","categorical_default_rates"]:
        p = PLOT_DIR / f"{n}.png"
        plots[n] = img_b64(p) if p.exists() else None
    for n in ["shap_summary_bar","shap_summary_dot","shap_waterfall_high_risk",
              "shap_waterfall_low_risk","shap_dependence_ext2"]:
        p = SHAP_DIR / f"{n}.png"
        plots[n] = img_b64(p) if p.exists() else None

    with open(PLOT_DIR / "summary_stats.json") as f: stats = json.load(f)

    return render_template("dashboard.html", user=session, best=best,
                           best_key=best_key, best_label=labels.get(best_key, best_key),
                           thresholds=thresholds, table=table, plots=plots, stats=stats)


@app.route("/predictor")
@login_required
def predictor():
    best_key  = (MODELS_DIR / "best_model.txt").read_text().strip()
    threshold = json.loads((MODELS_DIR / "thresholds.json").read_text()).get(best_key, 0.5)
    return render_template("predictor.html", user=session, threshold=threshold)


@app.route("/batch")
@login_required
def batch():
    return render_template("batch.html", user=session)


# ── prediction API ────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    art = get_artifacts()
    try:
        raw_df  = build_input_df(request.json)
        X_arr   = preprocess(raw_df, art["prep"])
        prob    = float(art["model"].predict_proba(X_arr)[0, 1])
        factors = get_shap_factors(art["explainer"], X_arr, art["feat_names"])
        shap_img= shap_waterfall_b64(art["explainer"], X_arr, art["feat_names"])
        label, cls, approved = risk_info(prob, art["threshold"])
        conf    = round(abs(prob - art["threshold"]) / max(art["threshold"], 1 - art["threshold"]) * 100, 1)
        return jsonify({"prob": round(prob * 100, 1), "risk_label": label,
                        "risk_class": cls, "approved": approved,
                        "threshold": round(art["threshold"] * 100, 1),
                        "confidence": conf, "top_factors": factors, "shap_img": shap_img})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_predict", methods=["POST"])
@login_required
def api_batch_predict():
    art = get_artifacts()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        df  = pd.read_csv(request.files["file"])
        # Normalise column names (lowercase, strip spaces)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        COL_DEFAULTS = {
            "age": 35, "gender": "Male", "education": "Secondary / secondary special",
            "family_status": "Married", "n_children": 0, "income": 300000,
            "loan_amt": 500000, "annuity": 17500, "own_car": 0, "own_realty": 1,
            "ext1": 0.5, "ext2": 0.5, "ext3": 0.5,
            "emp_type": "Employed", "years_employed": 5, "n_inquiries": 2,
        }
        for col, default in COL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        results = []
        for _, row in df.iterrows():
            raw = build_input_df(row.to_dict())
            X   = preprocess(raw, art["prep"])
            p   = float(art["model"].predict_proba(X)[0, 1])
            label, cls, approved = risk_info(p, art["threshold"])
            results.append({**row.to_dict(),
                             "default_prob": round(p * 100, 1),
                             "risk_label": label,
                             "risk_class": cls,
                             "decision": "APPROVE" if approved else "REVIEW"})

        return jsonify({"count": len(results), "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── AI endpoints ──────────────────────────────────────────────────────────────

@app.route("/api/ai/narrative", methods=["POST"])
@login_required
def api_narrative():
    from ai_analyst import generate_risk_narrative
    d = request.json
    return jsonify({"text": generate_risk_narrative(
        d["prob"] / 100, d["risk_label"], d["top_factors"], d["applicant_summary"])})


@app.route("/api/ai/improve", methods=["POST"])
@login_required
def api_improve():
    from ai_analyst import generate_improvement_plan
    d = request.json
    return jsonify({"text": generate_improvement_plan(
        d["prob"] / 100, d["top_factors"], d["applicant_summary"])})


@app.route("/api/ai/chat", methods=["POST"])
@login_required
def api_chat():
    from ai_analyst import chat_with_analyst
    d = request.json
    return jsonify({"reply": chat_with_analyst(
        d["question"], d["prob"] / 100, d["risk_label"],
        d["top_factors"], d["applicant_summary"], d.get("history", []))})


@app.route("/api/threshold_metrics")
@login_required
def api_threshold_metrics():
    """Return precision, recall, F1, approval_rate across threshold range for the slider."""
    try:
        preds_data = joblib.load(MODELS_DIR / "test_preds.pkl")
        probs  = preds_data["probs"]
        y_true = preds_data["y_true"]
        n_total = len(y_true)

        thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.01)]
        rows = []
        for t in thresholds:
            preds   = (probs >= t).astype(int)
            tp = int(((preds == 1) & (y_true == 1)).sum())
            fp = int(((preds == 1) & (y_true == 0)).sum())
            fn = int(((preds == 0) & (y_true == 1)).sum())
            prec    = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec     = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            approval_rate = round((preds == 0).sum() / n_total * 100, 1)
            rows.append({
                "threshold":     t,
                "precision":     round(prec, 4),
                "recall":        round(rec,  4),
                "f1":            round(f1,   4),
                "approval_rate": approval_rate,
            })
        return jsonify({"data": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai/protect", methods=["POST"])
@login_required
def api_protect():
    from ai_analyst import generate_protection_tips
    d = request.json
    return jsonify({"text": generate_protection_tips(
        d["prob"] / 100, d["risk_label"], d["top_factors"], d["applicant_summary"])})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
