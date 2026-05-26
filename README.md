# Credit Risk Scoring Model


End-to-end credit risk scoring system trained on **307,511 real Home Credit loan applications**. Predicts default probability with per-customer SHAP explanations and a Groq AI analyst.

---

## Results

| Model | ROC-AUC | PR-AUC | Recall (Defaults) |
|-------|---------|--------|-------------------|
| LightGBM | **0.766** | **0.252** | **42%** |
| XGBoost | 0.761 | 0.252 | 37% |
| Logistic Regression | 0.743 | 0.232 | 40% |

---

## Features

- **3 data sources joined**: `application_train.csv` + `bureau.csv` (credit history) + `bureau_balance.csv` (12.4M monthly payment records)
- **204 features** after engineering + OHE — bureau payment history in top 10 SHAP features
- **SMOTE** for 8% class imbalance
- **Per-customer SHAP** waterfall explanations
- **CIBIL score input** (300–900 range, normalized internally)
- **Batch CSV scoring** — upload multiple applicants, download results
- **Business threshold slider** — drag to see how approval rate, precision and recall change in real time
- **Groq Llama 3.3-70B** integration: risk narrative, safety guide, improvement plan, and chat analyst

---

## Stack

`Flask` · `LightGBM` · `XGBoost` · `SHAP` · `Groq API` · `scikit-learn` · `SMOTE` · `pandas` · `bcrypt`

---

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run pipeline (first time only)

```bash
# Place Home Credit CSVs in data/raw/ first
# Download from: https://www.kaggle.com/c/home-credit-default-risk/data

python src/generate_data.py     # detects real data automatically
python src/eda.py
python src/preprocessing.py
python src/train_models.py
python src/explainability.py
```

## Run app

```bash
# Set your Groq API key in .env:
# GROQ_API_KEY=your_key_here

python app.py
# Open http://localhost:5000
```

**Demo login:** `demo` / `demo123`

---

## Project Structure

```
credit_risk/
├── app.py                  # Flask app — routes, prediction API, AI endpoints
├── ai_analyst.py           # Groq AI — narrative, safety guide, improvement plan, chat
├── src/
│   ├── generate_data.py    # Detects real vs synthetic data
│   ├── eda.py              # EDA plots
│   ├── preprocessing.py    # Feature engineering + bureau joins + SMOTE
│   ├── train_models.py     # Trains LR + XGBoost + LightGBM, saves best
│   └── explainability.py  # SHAP global + per-customer plots
├── templates/              # HTML pages (login, signup, dashboard, predictor, batch)
├── static/css/style.css    # Custom CSS — no frameworks
├── data/raw/               # Home Credit CSVs (not committed — see .gitignore)
├── models/                 # Trained model artifacts
└── notebooks/eda_output/   # EDA + SHAP plots
```

---

## Demo Cases

| Case | CIBIL | Result |
|------|-------|--------|
| Best — senior female, 18 yrs employed | 860 | 3% · LOW RISK · APPROVE |
| Good — experienced, stable income | 775 | 3% · LOW RISK · APPROVE |
| Medium — average profile | 620 | 21% · LOW RISK · APPROVE |
| Bad — young, high debt, late payments | 430 | 46% · HIGH RISK · REVIEW |
| Worst — unemployed, 3 kids, 120+ DPD | 315 | 64% · HIGH RISK · REVIEW |
