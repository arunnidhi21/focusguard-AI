"""
FocusGuard AI — Training script
================================
Loads the logged Instagram usage data, engineers features, evaluates
the model with 5-fold cross-validation (not a single fragile split),
tunes hyperparameters with GridSearchCV, and saves the final model +
preprocessor + feature importances to disk for app.py to load.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "instagram_usage_data.csv"
MODEL_PATH = "focusguard_model.pkl"

FEATURE_COLUMNS = [
    "hour",
    "previous_app",
    "instagram_notification",
    "instagram_opens_30min",
    "instagram_opens_2hr",
    "previous_instagram_duration",
    "time_since_previous_instagram",
    "rapid_reopen",
    "day_of_week",
    "is_weekend",
]

CATEGORICAL_COLUMNS = ["previous_app"]


# ============================================================
# 1. LOAD + CLEAN
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M"
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["time_since_previous_instagram"] = (
        df["datetime"].diff().dt.total_seconds() / 60
    )
    df["time_since_previous_instagram"] = df[
        "time_since_previous_instagram"
    ].fillna(999)

    df["instagram_notification"] = df["instagram_notification"].map(
        {"yes": 1, "no": 0}
    )

    df["rapid_reopen"] = (df["time_since_previous_instagram"] <= 15).astype(int)

    # NOTE: "reason" and "wanted_to_stop" are deliberately excluded.
    # Both are recorded at the same time as the "intentional" label and
    # effectively restate it (e.g. reason="habit" almost always implies
    # intentional="no"). Using them as features would be leakage — the
    # model would look great and mean nothing.

    df["intentional"] = df["intentional"].map({"yes": 1, "no": 0})

    return df


# ============================================================
# 3. CROSS-VALIDATED EVALUATION (the honest number)
# ============================================================
def cross_validated_accuracy(X, y, preprocessor, param_grid, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pipe = Pipeline(
        [
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(random_state=42, class_weight="balanced")),
        ]
    )

    search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    search.fit(X, y)

    return search


# ============================================================
# 4. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("FOCUSGUARD AI — TRAINING")
    print("=" * 60)

    df = load_data(DATA_PATH)
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["intentional"]

    print(f"\nDataset samples: {len(df)}")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    majority_baseline = y.value_counts(normalize=True).max()
    print(f"Majority-class baseline: {majority_baseline * 100:.2f}%")

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)],
        remainder="passthrough",
    )

    # ---- Honest metric: 5-fold cross-validated accuracy over the
    # whole dataset, with hyperparameters chosen by grid search
    # inside that same CV loop. This is far more stable than a
    # single 80/20 split on ~290 rows, where one misclassified row
    # swings "accuracy" by nearly 2 points.
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 4, 5, 6, None],
        "clf__min_samples_leaf": [1, 2, 3, 5],
    }

    search = cross_validated_accuracy(X, y, preprocessor, param_grid)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"Best params: {search.best_params_}")
    print(f"Best 5-fold CV accuracy: {search.best_score_ * 100:.2f}%")
    print(
        f"Improvement over majority baseline: "
        f"{(search.best_score_ - majority_baseline) * 100:.2f} points"
    )

    # ---- Held-out chronological check, for a human-readable
    # confusion matrix / classification report (illustrative only —
    # the CV score above is the number to trust and quote).
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\n" + "=" * 60)
    print(f"CHRONOLOGICAL HOLD-OUT CHECK (last {len(X_test)} sessions)")
    print("=" * 60)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(
        "NOTE: this hold-out slice is small — treat it as a sanity check, "
        "not the headline number."
    )

    # ---- Refit the best model + preprocessor on ALL data for the
    # deployed artifact (the GUI should use every logged session).
    final_pipeline = search.best_estimator_
    final_pipeline.fit(X, y)

    fitted_preprocessor = final_pipeline.named_steps["pre"]
    fitted_model = final_pipeline.named_steps["clf"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_model.feature_importances_
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (final model, trained on all data)")
    print("=" * 60)
    print(feature_importance.head(10).to_string(index=False))

    # ---- Save everything app.py needs.
    joblib.dump(
        {
            "preprocessor": fitted_preprocessor,
            "model": fitted_model,
            "feature_columns": FEATURE_COLUMNS,
            "feature_importance": feature_importance,
            "cv_accuracy": search.best_score_,
            "majority_baseline": majority_baseline,
            "n_samples": len(df),
        },
        MODEL_PATH,
    )

    print(f"\nSaved trained model + preprocessor to {MODEL_PATH}")
    print("Run app.py to launch the GUI using this saved model.")


if __name__ == "__main__":
    main()
