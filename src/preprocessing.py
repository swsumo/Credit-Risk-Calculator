import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH          = Path("data/raw/application_train.csv")
BUREAU_PATH        = Path("data/raw/bureau.csv")
BUREAU_BAL_PATH    = Path("data/raw/bureau_balance.csv")
PROCESSED_DIR      = Path("data/processed")
MODELS_DIR         = Path("models")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# STATUS codes: 0=no DPD, 1=1-30d, 2=31-60d, 3=61-90d, 4=91-120d, 5=120+d, C=closed, X=unknown
STATUS_TO_DPD = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "C": 0, "X": 0}


def build_bureau_balance_features(bb_path: Path, bureau_path: Path) -> pd.DataFrame:
    """
    Aggregate bureau_balance.csv (monthly payment history) into per-applicant features.
    Two-level join: bureau_balance → per credit (SK_ID_BUREAU) → per applicant (SK_ID_CURR).
    """
    print("Loading bureau_balance.csv…")
    bb = pd.read_csv(bb_path)
    bb["DPD"] = bb["STATUS"].map(STATUS_TO_DPD).fillna(0).astype(int)

    # Level 1: aggregate per credit (SK_ID_BUREAU)
    per_credit = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT   = ("MONTHS_BALANCE", "count"),
        BB_MAX_DPD        = ("DPD",            "max"),
        BB_MEAN_DPD       = ("DPD",            "mean"),
        BB_LATE_COUNT     = ("DPD",            lambda x: (x > 0).sum()),
        BB_SEVERE_COUNT   = ("DPD",            lambda x: (x >= 3).sum()),  # 61+ days
        BB_LATE_RECENT    = ("DPD",            lambda x: (x[bb.loc[x.index, "MONTHS_BALANCE"] >= -12] > 0).sum()),
    ).reset_index()

    # Level 2: link credits → applicants via bureau.csv, then aggregate to SK_ID_CURR
    print("Linking bureau_balance → SK_ID_CURR…")
    bureau_ids = pd.read_csv(bureau_path, usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
    merged = bureau_ids.merge(per_credit, on="SK_ID_BUREAU", how="left")

    per_app = merged.groupby("SK_ID_CURR").agg(
        BB_MAX_DPD_MAX      = ("BB_MAX_DPD",      "max"),   # worst single DPD category ever
        BB_MEAN_DPD_MEAN    = ("BB_MEAN_DPD",     "mean"),  # average lateness across all credits
        BB_LATE_COUNT_SUM   = ("BB_LATE_COUNT",   "sum"),   # total late months across all credits
        BB_SEVERE_COUNT_SUM = ("BB_SEVERE_COUNT", "sum"),   # total 61+ day late months
        BB_LATE_RECENT_SUM  = ("BB_LATE_RECENT",  "sum"),   # late months in last 12 months
        BB_MONTHS_TOTAL     = ("BB_MONTHS_COUNT", "sum"),   # total months of credit history
    ).reset_index()

    print(f"Bureau balance aggregated: {len(per_app):,} applicants, {per_app.shape[1]-1} features")
    return per_app


def build_bureau_features(bureau_path: Path) -> pd.DataFrame:
    """Aggregate bureau.csv into one row per SK_ID_CURR."""
    print("Loading bureau.csv…")
    b = pd.read_csv(bureau_path, usecols=[
        "SK_ID_CURR", "CREDIT_ACTIVE", "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE",
        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
        "CNT_CREDIT_PROLONG",
    ])
    agg = b.groupby("SK_ID_CURR").agg(
        BUREAU_CNT               = ("SK_ID_CURR",          "count"),
        BUREAU_CNT_ACTIVE        = ("CREDIT_ACTIVE",       lambda x: (x == "Active").sum()),
        BUREAU_DAYS_CREDIT_MEAN  = ("DAYS_CREDIT",         "mean"),
        BUREAU_OVERDUE_MAX       = ("CREDIT_DAY_OVERDUE",  "max"),
        BUREAU_AMT_OVERDUE_SUM   = ("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_AMT_DEBT_SUM      = ("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_MAX_OVERDUE_EVER  = ("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_PROLONG_SUM       = ("CNT_CREDIT_PROLONG",  "sum"),
    ).reset_index()
    print(f"Bureau aggregated: {len(agg):,} applicants, {agg.shape[1]-1} features")
    return agg


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYMENT_YEARS"] = np.where(df["DAYS_EMPLOYED"] == 365243, -1, -df["DAYS_EMPLOYED"] / 365)
    df["IS_UNEMPLOYED"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)

    # EXT_SOURCE_MEAN / MIN from whichever bureau score columns survived the >40% drop
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_MIN"]  = df[ext_cols].min(axis=1)
    # Do NOT re-add dropped columns — let them stay absent so the pipeline is clean
    return df


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Join bureau aggregate features (credit counts, debt, overdue amounts)
    if BUREAU_PATH.exists():
        bureau_agg = build_bureau_features(BUREAU_PATH)
        df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
        print(f"After bureau merge: {df.shape}")
    else:
        print("bureau.csv not found — skipping")

    # Join bureau_balance payment history features (most valuable signal)
    if BUREAU_BAL_PATH.exists() and BUREAU_PATH.exists():
        bb_agg = build_bureau_balance_features(BUREAU_BAL_PATH, BUREAU_PATH)
        df = df.merge(bb_agg, on="SK_ID_CURR", how="left")
        print(f"After bureau_balance merge: {df.shape}")
    else:
        print("bureau_balance.csv not found — skipping")

    # Drop high-missing columns (>40%)
    missing_rate = df.isnull().mean()
    drop_cols = missing_rate[missing_rate > 0.4].index.tolist()
    if drop_cols:
        print(f"Dropping columns with >40% missing: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    # Feature engineering
    print("Engineering features...")
    df = engineer_features(df)

    X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    y = df["TARGET"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Numeric cols: {len(numeric_cols)}, Categorical cols: {len(categorical_cols)}")

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    # Stratified 70/15/15 split
    X_train_raw, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    print("Fitting preprocessor...")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Feature names — derive from the fitted pipeline (handles any dropped columns)
    num_out = preprocessor.named_transformers_["num"].get_feature_names_out(numeric_cols).tolist()
    ohe     = preprocessor.named_transformers_["cat"]["ohe"]
    cat_out = ohe.get_feature_names_out(categorical_cols).tolist()
    feature_names = num_out + cat_out
    assert len(feature_names) == X_train.shape[1], \
        f"Feature name mismatch: {len(feature_names)} names vs {X_train.shape[1]} columns"
    print(f"Total features after OHE: {len(feature_names)}")

    # SMOTE on training set only
    print(f"\nClass distribution before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    smote = SMOTE(sampling_strategy=0.33, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE:  {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Save artifacts
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    joblib.dump(X_train, PROCESSED_DIR / "X_train.pkl")
    joblib.dump(X_val, PROCESSED_DIR / "X_val.pkl")
    joblib.dump(X_test, PROCESSED_DIR / "X_test.pkl")
    joblib.dump(y_train.values if hasattr(y_train, "values") else y_train, PROCESSED_DIR / "y_train.pkl")
    joblib.dump(y_val.values if hasattr(y_val, "values") else y_val, PROCESSED_DIR / "y_val.pkl")
    joblib.dump(y_test.values if hasattr(y_test, "values") else y_test, PROCESSED_DIR / "y_test.pkl")

    print("\nFinal shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {len(y_train)}")
    print(f"  X_val:   {X_val.shape},   y_val:   {len(y_val)}")
    print(f"  X_test:  {X_test.shape},  y_test:  {len(y_test)}")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
