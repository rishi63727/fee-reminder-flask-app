#!/usr/bin/env python3
"""
Fee Payment Risk Prediction Model - Historical Data Training (Full)
- Robust schema normalization (aligned with app.py)
- Data structure analysis & target auto-detection
- Multiple models with CV AUC comparison
- Hyperparameter tuning on best baseline
- Pipeline with ColumnTransformer (no manual get_dummies)
- Saves model + metadata (+ optional feature importance plot)
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Optional plotting (will skip if not available in headless env)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fee-train")

# -------------------------------------------------------------------
# Schema normalization (MUST MATCH app.py)
# -------------------------------------------------------------------
ALIASES = {
    # features
    "paymentplan": "payment_plan",
    "has_payment_plan": "payment_plan",
    "scholarship": "scholarship",
    "has_scholarship": "scholarship",
    "pastlatepayments": "past_late_payments",
    "previous_late_payments": "past_late_payments",
    # target
    "late_payment": "late_payment",
    "waslate": "late_payment",
    "is_late_payment": "late_payment",
    # identifiers/optional
    "studentid": "student_id",
    "student_id": "student_id",
    "studentname": "student_name",
}

def to_snake(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return name.strip().lower().replace(" ", "_")

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Convert headers to snake_case and apply aliases + value coercions."""
    df = df.copy()
    df.columns = [to_snake(c) for c in df.columns]
    df.rename(columns={c: ALIASES.get(c, c) for c in df.columns}, inplace=True)

    if "payment_plan" in df.columns:
        df["payment_plan"] = df["payment_plan"].astype(str).str.strip().str.lower()

    if "scholarship" in df.columns:
        df["scholarship"] = (
            df["scholarship"].astype(str).str.strip().str.lower()
            .map({"y": "yes", "yes": "yes", "1": "yes", "true": "yes",
                  "n": "no", "no": "no", "0": "no", "false": "no"})
            .fillna("no")
        )

    if "past_late_payments" in df.columns:
        df["past_late_payments"] = pd.to_numeric(df["past_late_payments"], errors="coerce").fillna(0).astype(int)

    # Dates (useful if your CSV/Excel has them)
    for date_col in ["last_payment_date", "due_date", "actual_payment_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    return df

# -------------------------------------------------------------------
# Analysis helpers
# -------------------------------------------------------------------
def analyze_structure(df: pd.DataFrame) -> List[str]:
    print("\n" + "=" * 60)
    print("DATA STRUCTURE ANALYSIS")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumn Information:")
    print("-" * 40)
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) else 0
        unique_count = df[col].nunique()
        print(f"{i+1:2d}. {col:25} | {str(dtype):12} | Nulls: {null_count:5d} ({null_pct:5.1f}%) | Unique: {unique_count:6d}")

    print("\nSample Data:")
    print("-" * 40)
    print(df.head())

    print("\nPotential target columns (late/overdue/default/missed/status/paid):")
    print("-" * 40)
    candidates = []
    for col in df.columns:
        c = col.lower()
        if any(k in c for k in ["late", "overdue", "delay", "default", "missed", "status", "paid"]):
            candidates.append(col)
            uniq = df[col].dropna().unique()
            print(f"- {col}: {uniq[:10]}")
    if not candidates:
        print("(None found)")
    return candidates

def auto_target(df: pd.DataFrame) -> Optional[pd.Series]:
    """Try to auto-detect an existing binary 'late_payment' style column."""
    # 1) Exact match
    if "late_payment" in df.columns:
        y = df["late_payment"]
        return to_binary(y)

    # 2) Any 0/1/bool column with 'late/overdue/default/missed' keyword
    for col in df.columns:
        c = col.lower()
        if any(k in c for k in ["late", "overdue", "delay", "default", "missed"]):
            y = to_binary(df[col])
            if y is not None and y.nunique(dropna=True) == 2:
                print(f"Auto-detected target: {col} -> late_payment")
                return y

    return None

def to_binary(series: pd.Series) -> Optional[pd.Series]:
    """Map a series to 0/1 if possible."""
    s = series.copy()
    if s.dtype == bool:
        return s.astype(int)

    if s.dtype == object:
        # Map common strings
        m = {"yes": 1, "true": 1, "1": 1, "late": 1, "overdue": 1, "defaulted": 1,
             "no": 0, "false": 0, "0": 0, "on_time": 0, "ontime": 0, "paid": 0}
        ss = s.astype(str).str.strip().str.lower().map(m)
        # If mapping worked for most rows, fill the rest if numeric
        if ss.notna().mean() > 0.6:
            ss = ss.where(ss.notna(), pd.to_numeric(s, errors="coerce"))
            return ss.fillna(0).astype(int)

    # numeric?
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().any():
        uniq = set(sn.dropna().unique().tolist())
        if uniq.issubset({0, 1}):
            return sn.astype(int)

    return None

def build_pipeline(clf) -> Pipeline:
    """Create a unified pipeline preprocessor + classifier."""
    cat_cols = ["payment_plan", "scholarship"]
    num_cols = ["past_late_payments"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([("pre", pre), ("clf", clf)])

def cv_auc(model: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> Tuple[float, float]:
    scores = cross_val_score(model, X, y, cv=folds, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train_full(input_path: str, output_path: str):
    # Load
    log.info("Loading data from %s", input_path)
    if input_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    # Normalize schema
    df = normalize_schema(df)

    # Analysis
    analyze_structure(df)

    # Target
    y = auto_target(df)
    if y is None:
        # try from dates if both exist
        if {"actual_payment_date", "due_date"}.issubset(df.columns):
            print("\nCreating target from payment vs due dates (>7 days late → 1)")
            days_late = (df["actual_payment_date"] - df["due_date"]).dt.days
            y = (days_late > 7).astype(int)
        else:
            raise ValueError(
                "Could not auto-detect target. Provide a late indicator column "
                "or include actual_payment_date & due_date for derivation."
            )

    # Features (IMPORTANT: keep ONLY these three to match the app!)
    FEATURES = ["payment_plan", "scholarship", "past_late_payments"]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}. Found: {list(df.columns)}")

    X = df[FEATURES].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=0.2, random_state=42, stratify=y
    )

    # Baseline models
    baselines = {
        "LogisticRegression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        ),
        "RandomForest": build_pipeline(
            RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        ),
        "GradientBoosting": build_pipeline(
            GradientBoostingClassifier(random_state=42)
        ),
    }

    results = {}
    best_name, best_model, best_auc = None, None, -1.0

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON (CV AUC, 5-fold)")
    print("=" * 60)
    for name, pipe in baselines.items():
        mean_auc, std_auc = cv_auc(pipe, X_train, y_train, folds=5)
        results[name] = (mean_auc, std_auc)
        print(f"- {name:18s}: AUC {mean_auc:.4f} (+/- {std_auc:.4f})")
        if mean_auc > best_auc:
            best_auc, best_name, best_model = mean_auc, name, pipe

    print(f"\nBest baseline: {best_name} (CV AUC={best_auc:.4f})")

    # Hyperparameter tuning on best model
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 60)

    # Choose param grid depending on classifier type
    base_clf = best_model.named_steps["clf"]
    if isinstance(base_clf, LogisticRegression):
        param_grid = {
            "clf__C": [0.1, 1, 5, 10],
            "clf__penalty": ["l2"],  # liblinear not used; stick to lbfgs with l2
        }
    elif isinstance(base_clf, RandomForestClassifier):
        param_grid = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        }
    else:  # GradientBoostingClassifier
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5],
        }

    grid = GridSearchCV(
        best_model,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    tuned = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV AUC: {grid.best_score_:.4f}")

    # Final evaluation
    y_proba = tuned.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    final_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION (Holdout Test)")
    print("=" * 60)
    print(f"AUC: {final_auc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model (Pipeline) + metadata
    joblib.dump(tuned, output_path)
    log.info("Saved model to %s", output_path)

    metadata = {
        "model_path": output_path,
        "trained_at": datetime.now().isoformat(),
        "features": FEATURES,
        "classifier": type(tuned.named_steps["clf"]).__name__,
        "best_params": grid.best_params_,
        "cv_auc": grid.best_score_,
        "test_auc": float(final_auc),
        "input": os.path.abspath(input_path),
        "notes": "Pipeline includes ColumnTransformer + OneHotEncoder; app must pass raw features only.",
    }
    with open("historical_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved metadata to historical_model_metadata.json")

    # Optional: quick feature importance bar for tree models
    if PLOTTING_AVAILABLE and isinstance(tuned.named_steps["clf"], RandomForestClassifier):
        try:
            # Get transformed feature names from preprocessor
            pre: ColumnTransformer = tuned.named_steps["pre"]
            ohe: OneHotEncoder = pre.named_transformers_["cat"]
            cat_cols = ["payment_plan", "scholarship"]
            num_cols = ["past_late_payments"]

            ohe_feat = ohe.get_feature_names_out(cat_cols).tolist()
            feat_names = ohe_feat + num_cols

            importances = tuned.named_steps["clf"].feature_importances_
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
                "importance", ascending=False
            )

            top = imp_df.head(15)
            plt.figure(figsize=(10, 7))
            sns.barplot(data=top, x="importance", y="feature")
            plt.title("Feature Importance (RandomForest)")
            plt.tight_layout()
            plt.savefig("historical_feature_importance.png", dpi=300, bbox_inches="tight")
            log.info("Saved feature importance plot to historical_feature_importance.png")
        except Exception as e:
            log.warning("Could not generate feature importance plot: %s", e)

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train fee risk model on historical data.")
    p.add_argument("--input", "-i", default="historical_data.xlsx",
                   help="Training data path (.xlsx/.xls/.csv). Default: historical_data.xlsx")
    p.add_argument("--output", "-o", default="fee_prediction_model.pkl",
                   help="Output model path. Default: fee_prediction_model.pkl")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    in_path = args.input
    out_path = args.output

    if not Path(in_path).exists():
        raise FileNotFoundError(f"Training file not found: {in_path}")

    train_full(in_path, out_path)
