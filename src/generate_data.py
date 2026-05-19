"""
Data loader / generator for Credit Risk project.

Priority:
  1. If data/raw/application_train.csv already has >100K rows → treat as real Home Credit data, skip.
  2. Otherwise → generate 50K synthetic rows and save there.

Real Home Credit data: download application_train.csv from
  https://www.kaggle.com/c/home-credit-default-risk/data
  and place it at data/raw/application_train.csv, then re-run:
    python src/preprocessing.py
    python src/train_models.py
    python src/explainability.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path("data/raw/application_train.csv")
REAL_DATA_THRESHOLD = 100_000


def is_real_data() -> bool:
    if not OUTPUT_PATH.exists():
        return False
    with open(OUTPUT_PATH) as f:
        count = sum(1 for _ in f) - 1
    return count >= REAL_DATA_THRESHOLD


def generate_synthetic(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sk_ids = np.arange(100002, 100002 + n)

    amt_credit  = np.clip(rng.lognormal(np.log(500_000), 0.6, n), 45_000, 4_050_000)
    amt_income  = np.clip(rng.lognormal(np.log(180_000), 0.55, n), 25_000, 1_500_000)
    amt_annuity = amt_credit * rng.uniform(0.025, 0.05, n)
    amt_goods   = amt_credit * rng.uniform(0.8, 1.0, n)

    days_birth    = -rng.integers(7_300, 25_550, n)
    unemployed    = rng.random(n) < 0.28
    days_employed = np.where(unemployed, 365_243, -rng.integers(1, 15_000, n))
    days_reg      = -rng.integers(1, 25_000, n)

    cnt_children = rng.choice([0,1,2,3,4,5], n, p=[0.55,0.25,0.12,0.05,0.02,0.01])
    cnt_fam      = cnt_children + rng.choice([1,2], n, p=[0.3,0.7])

    ext1 = rng.beta(3, 3, n).astype(float); ext1[rng.random(n) < 0.15] = np.nan
    ext2 = rng.beta(3, 3, n)
    ext3 = rng.beta(3, 3, n)

    amt_req = rng.poisson(3, n).astype(float); amt_req[rng.random(n) < 0.20] = np.nan
    obs_30  = rng.integers(0, 11, n).astype(float); obs_30[rng.random(n) < 0.05] = np.nan
    def_30  = rng.integers(0, 6,  n).astype(float); def_30[rng.random(n) < 0.05] = np.nan

    contract    = rng.choice(["Cash loans","Revolving loans"], n, p=[0.90,0.10])
    gender      = rng.choice(["M","F"], n, p=[0.56,0.44])
    flag_car    = rng.choice(["Y","N"], n, p=[0.34,0.66])
    flag_real   = rng.choice(["Y","N"], n, p=[0.69,0.31])
    education   = rng.choice(["Secondary / secondary special","Higher education",
                               "Incomplete higher","Lower secondary"], n, p=[0.71,0.24,0.04,0.01])
    fam_status  = rng.choice(["Married","Single / not married","Civil marriage","Separated","Widow"],
                              n, p=[0.64,0.14,0.10,0.07,0.05])
    income_type = rng.choice(["Working","Commercial associate","Pensioner","State servant"],
                              n, p=[0.52,0.23,0.18,0.07])
    housing     = rng.choice(["House / apartment","With parents","Municipal apartment",
                               "Rented apartment","Office apartment"], n, p=[0.88,0.05,0.04,0.02,0.01])
    orgs = ["Business Entity Type 3","School","Government","Religion","Other","Medicine",
            "Business Entity Type 2","Self-employed","Transport: type 2","Construction",
            "Housing","Kindergarten","Trade: type 7","Industry: type 11","Military",
            "Services","Security Ministries","Transport: type 4","Industry: type 1","Emergency"]
    org_w = [0.12,0.10,0.08,0.07,0.07,0.06,0.06,0.05,0.05,0.05,0.04,0.04,0.04,
             0.03,0.03,0.03,0.03,0.02,0.02,0.01]
    organization = rng.choice(orgs, n, p=org_w)

    p = np.full(n, 0.075)
    p[unemployed] *= 2.0
    p[ext2 < 0.3] *= 3.0
    p[(amt_credit / (amt_income + 1)) > 10] *= 2.5
    p[cnt_children >= 3] *= 1.5
    p[gender == "M"] *= 1.3
    p = np.clip(p, 0, 0.90)
    target = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "SK_ID_CURR": sk_ids, "TARGET": target,
        "NAME_CONTRACT_TYPE": contract, "CODE_GENDER": gender,
        "FLAG_OWN_CAR": flag_car, "FLAG_OWN_REALTY": flag_real,
        "CNT_CHILDREN": cnt_children, "AMT_INCOME_TOTAL": amt_income,
        "AMT_CREDIT": amt_credit, "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods, "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": fam_status, "NAME_INCOME_TYPE": income_type,
        "NAME_HOUSING_TYPE": housing, "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed, "DAYS_REGISTRATION": days_reg,
        "CNT_FAM_MEMBERS": cnt_fam, "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        "AMT_REQ_CREDIT_BUREAU_YEAR": amt_req,
        "OBS_30_CNT_SOCIAL_CIRCLE": obs_30, "DEF_30_CNT_SOCIAL_CIRCLE": def_30,
        "ORGANIZATION_TYPE": organization,
    })


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if is_real_data():
        n_rows = sum(1 for _ in open(OUTPUT_PATH)) - 1
        print(f"Real Home Credit data detected ({n_rows:,} rows). Skipping generation.")
    elif OUTPUT_PATH.exists():
        n_rows = sum(1 for _ in open(OUTPUT_PATH)) - 1
        print(f"Synthetic data already exists ({n_rows:,} rows). Skipping.")
    else:
        print("Generating synthetic dataset (50K rows)…")
        df = generate_synthetic()
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved → {OUTPUT_PATH}")

    df = pd.read_csv(OUTPUT_PATH)
    print(f"Shape: {df.shape}  |  Default rate: {df['TARGET'].mean():.2%}")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("Null counts:", null_counts[null_counts > 0].to_dict())
