"""
Step 4 - Model building & training (OULAD early-warning project)

Input : data/model_data.csv produced in step 2
        data/raw/studentRegistration.csv (raw OULAD file, kept local, not in git)
Task  : binary classification
        at_risk = 1  -> final_result in {Fail, Withdrawn}
        at_risk = 0  -> final_result in {Pass, Distinction}
Models: Logistic Regression, Decision Tree, Random Forest

Outputs (written next to this script):
    models/<model>.joblib            full sklearn Pipeline (preprocessing + model)
    outputs/split.csv                which rows went to train / test
    outputs/test_predictions.csv     predictions + probabilities on the test set
    outputs/model_comparison.csv     test metrics + best CV params per model
    outputs/feature_importance.csv   importances (trees) / coefficients (LR)
    outputs/run_info.json            features, target, filters, sizes, versions

Run:
    python train_models.py
    python train_models.py --data path/to/model_data.csv --registration path/to/studentRegistration.csv
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent

# Prediction is made at the end of day 27 (first 28 days: day 0..27).
PREDICTION_DAY = 28

KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
TARGET_SOURCE = "final_result"
TARGET = "at_risk"
POSITIVE_RESULTS = ["Fail", "Withdrawn"]

# --------------------------------------------------------------------------
# Allowed features: only information known at day 28 of the course.
# --------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "code_module",
    "presentation_start",   # B = February start, J = October start (derived)
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]
NUMERIC_FEATURES = [
    "num_of_prev_attempts",
    "studied_credits",
    "total_clicks_first_28_days",
    "active_days_first_28_days",
    "unique_sites_first_28_days",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Never allowed as model input (identifiers, the label, or anything after day 28).
FORBIDDEN_FEATURES = {
    "id_student",
    "final_result",
    TARGET,
    "date_unregistration",
    "score",
    "date_submitted",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train LR / DT / RF on OULAD early activity.")
    p.add_argument(
        "--data",
        type=Path,
        default=HERE.parent / "data" / "model_data.csv",
        help="model_data.csv from step 2",
    )
    p.add_argument(
        "--registration",
        type=Path,
        default=HERE.parent / "data" / "raw" / "studentRegistration.csv",
        help="raw studentRegistration.csv, used only to drop students who had "
             "already withdrawn before the prediction day",
    )
    p.add_argument(
        "--keep-early-withdrawals",
        action="store_true",
        help="do NOT drop students who unregistered before day 28",
    )
    p.add_argument("--test-folds", type=int, default=5,
                   help="1/N of the data is held out as test (default 5 -> 20%%)")
    p.add_argument("--cv-folds", type=int, default=3,
                   help="folds for hyper-parameter search on the training set")
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(args):
    df = pd.read_csv(args.data)
    print(f"Loaded {args.data}  shape={df.shape}")

    missing = [c for c in KEY_COLUMNS + [TARGET_SOURCE] + NUMERIC_FEATURES
               + [c for c in CATEGORICAL_FEATURES if c != "presentation_start"]
               if c not in df.columns]
    if missing:
        sys.exit(f"model_data.csv is missing columns: {missing}")

    dupes = df.duplicated(subset=KEY_COLUMNS).sum()
    if dupes:
        sys.exit(f"{dupes} duplicated (module, presentation, student) rows - fix step 2 first")

    filter_info = {"early_withdrawal_filter": "not applied", "rows_removed": 0}

    if not args.keep_early_withdrawals:
        if args.registration.exists():
            reg = pd.read_csv(args.registration,
                              usecols=KEY_COLUMNS + ["date_unregistration"])
            df = df.merge(reg, on=KEY_COLUMNS, how="left", validate="one_to_one")
            # Students who left before the prediction day are not "to be predicted":
            # we already know they withdrew, and their (near) zero activity would
            # make the task look easier than it really is.
            early = df["date_unregistration"].notna() & (df["date_unregistration"] < PREDICTION_DAY)
            n_early = int(early.sum())
            df = df.loc[~early].drop(columns="date_unregistration").reset_index(drop=True)
            filter_info = {
                "early_withdrawal_filter": f"dropped date_unregistration < {PREDICTION_DAY}",
                "rows_removed": n_early,
            }
            print(f"Dropped {n_early} students who unregistered before day {PREDICTION_DAY}")
        else:
            print(f"WARNING: {args.registration} not found - early withdrawals NOT removed")
            filter_info["early_withdrawal_filter"] = "skipped (registration file not found)"
    else:
        filter_info["early_withdrawal_filter"] = "disabled by --keep-early-withdrawals"

    # Derived feature: start month of the presentation (B=Feb, J=Oct).
    # The year itself is left out so the model does not memorise a specific cohort.
    df["presentation_start"] = df["code_presentation"].str[-1]

    df[TARGET] = df[TARGET_SOURCE].isin(POSITIVE_RESULTS).astype(int)

    print("\nTarget distribution (at_risk):")
    print(df[TARGET].value_counts().rename({0: "0 Pass/Distinction", 1: "1 Fail/Withdrawn"}))
    print((df[TARGET].value_counts(normalize=True) * 100).round(2))
    return df, filter_info


def check_features():
    leaked = FORBIDDEN_FEATURES.intersection(FEATURES)
    assert not leaked, f"Forbidden columns used as features: {leaked}"
    assert len(set(FEATURES)) == len(FEATURES), "duplicated feature names"


def split_data(df, n_folds):
    """Hold out 1/n_folds as test. Stratified on the target and grouped by
    student, so the same student never appears in both train and test."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df[TARGET], groups=df["id_student"]))

    overlap = set(df.loc[train_idx, "id_student"]) & set(df.loc[test_idx, "id_student"])
    assert not overlap, f"{len(overlap)} students in both train and test"
    return train_idx, test_idx


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_preprocessor(scale_numeric):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def model_specs():
    """name -> (pipeline, parameter grid). class_weight='balanced' because the
    two classes are not equal in size."""
    return {
        "logistic_regression": (
            Pipeline([
                ("prep", make_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=RANDOM_STATE)),
            ]),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "decision_tree": (
            Pipeline([
                ("prep", make_preprocessor(scale_numeric=False)),
                ("model", DecisionTreeClassifier(class_weight="balanced",
                                                 random_state=RANDOM_STATE)),
            ]),
            {
                "model__max_depth": [4, 6, 8, 12, None],
                "model__min_samples_leaf": [1, 20, 50, 100],
            },
        ),
        "random_forest": (
            Pipeline([
                ("prep", make_preprocessor(scale_numeric=False)),
                ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=RANDOM_STATE, n_jobs=1)),
            ]),
            {
                "model__max_depth": [10, 20, None],
                "model__min_samples_leaf": [1, 5, 20],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
    }


def feature_names(fitted_pipe):
    return fitted_pipe.named_steps["prep"].get_feature_names_out()


def importance_table(name, fitted_pipe):
    model = fitted_pipe.named_steps["model"]
    names = feature_names(fitted_pipe)
    if hasattr(model, "feature_importances_"):
        values, kind = model.feature_importances_, "importance"
    else:
        values, kind = model.coef_.ravel(), "coefficient"
    out = pd.DataFrame({"model": name, "feature": names, "type": kind, "value": values})
    out["abs_value"] = out["value"].abs()
    return out.sort_values("abs_value", ascending=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    check_features()

    models_dir = HERE / "models"
    out_dir = HERE / "outputs"
    models_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    df, filter_info = load_data(args)
    train_idx, test_idx = split_data(df, args.test_folds)

    X_train, y_train = df.loc[train_idx, FEATURES], df.loc[train_idx, TARGET]
    X_test, y_test = df.loc[test_idx, FEATURES], df.loc[test_idx, TARGET]
    groups_train = df.loc[train_idx, "id_student"]
    print(f"\nTrain: {len(X_train)} rows  (at_risk={y_train.mean():.3f})")
    print(f"Test : {len(X_test)} rows  (at_risk={y_test.mean():.3f})")

    split = df[KEY_COLUMNS].copy()
    split["split"] = "train"
    split.loc[test_idx, "split"] = "test"
    split.to_csv(out_dir / "split.csv", index=False)

    predictions = df.loc[test_idx, KEY_COLUMNS + [TARGET_SOURCE, TARGET]].copy()
    predictions = predictions.rename(columns={TARGET: "y_true"}).reset_index(drop=True)

    cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results, importances = [], []

    for name, (pipe, grid) in model_specs().items():
        print(f"\n=== {name} ===")
        # Tuning uses the training set only; the test set is touched once below.
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=args.n_jobs, refit=True)
        search.fit(X_train, y_train, groups=groups_train)
        best = search.best_estimator_
        print(f"best params : {search.best_params_}")
        print(f"best CV F1  : {search.best_score_:.4f}")

        proba = best.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        predictions[f"{name}_pred"] = pred
        predictions[f"{name}_proba"] = proba.round(6)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "train_f1": f1_score(y_train, best.predict(X_train), zero_division=0),
            "cv_f1": search.best_score_,
            "best_params": json.dumps(search.best_params_, default=str),
        }
        results.append(row)
        print("test  : " + "  ".join(f"{k}={row[k]:.4f}" for k in
                                     ["accuracy", "precision", "recall", "f1", "roc_auc"]))

        joblib.dump(best, models_dir / f"{name}.joblib")
        importances.append(importance_table(name, best))

    comparison = pd.DataFrame(results).sort_values("f1", ascending=False)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False, float_format="%.4f")
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.concat(importances).to_csv(out_dir / "feature_importance.csv", index=False,
                                  float_format="%.6f")

    run_info = {
        "data_file": os.path.relpath(args.data, HERE),
        "rows_used": int(len(df)),
        "target": {"name": TARGET, "positive_class_1": POSITIVE_RESULTS,
                   "negative_class_0": ["Pass", "Distinction"]},
        "prediction_day": PREDICTION_DAY,
        **filter_info,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "excluded_columns": sorted(FORBIDDEN_FEATURES | {"code_presentation (year)"}),
        "split": {"method": "StratifiedGroupKFold (grouped by id_student), 1 fold held out",
                  "test_fraction": round(len(test_idx) / len(df), 4),
                  "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
                  "random_state": RANDOM_STATE},
        "tuning": {"method": "GridSearchCV on training set only", "scoring": "f1",
                   "cv_folds": args.cv_folds},
        "decision_threshold": 0.5,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print("\n=== Model comparison (test set) ===")
    print(comparison.drop(columns="best_params").to_string(index=False, float_format="%.4f"))
    print(f"\nSaved models to {models_dir}")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
