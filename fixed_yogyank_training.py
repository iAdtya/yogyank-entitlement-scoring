"""
Yogyank Entitlement Score - Baseline Training Script (Draft v1)
Author: Junior Data Scientist
Notes: Model is performing well. Validation score looks good. Ready for production.
"""

import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_and_prep_data(path="farmer_scoring_sample_yogyank_round1.csv"):
    return pd.read_csv(path)


# PM-Kisan Gov policy
# can change without retraining.
PM_KISAN_PENALTY = 150.0


def apply_policy(base_scores, pm_kisan_status):
    """Apply the PM-Kisan rule on top of the model's base score."""
    final = base_scores.copy()
    final[pm_kisan_status.values == "No"] -= PM_KISAN_PENALTY
    return final


def reason_codes(row, importances, medians, top_n=3):
    """Plain-English reasons for one farmer's score: the top numeric features,
    and whether the farmer is above/below the training median on each.
    Deterministic and dependency-free, so the reasons are stable across runs."""
    reasons = []
    for feat in [f for f in importances if f in medians.index][:top_n]:
        val = row[feat]
        if pd.isna(val):
            reasons.append(f"{feat}=missing (imputed)")
        else:
            side = "above" if val >= medians[feat] else "below"
            reasons.append(f"{feat}={val:.2f} ({side} median {medians[feat]:.2f})")
    return reasons


def train_model():
    df = load_and_prep_data()

    # The PM Kisan business policy should be applied after the model is trained to avoid retraining if the policy changes
    # print("Applying PM Kisan business policy...")
    # df.loc[df["pm_kisan_status"] == "No", "target_entitlement_score"] -= 150

    # Features assumed known at/before the scoring (application) date.
    numeric_cols = [
        "land_area_acres",
        "historical_repayment_score",
        "liability_ratio_pct",
        "annual_income_inr",
        "rainfall_deviation_pct",
        "ndvi_score",
    ]
    categorical_cols = [
        "crop_type",
        "district",
        "irrigation_type",
        "land_ownership",
        "soil_type",
        "sales_channel",
    ]
    features = numeric_cols + categorical_cols

    # "pm_kisan_status" is NOT a model feature — it is a policy input, applied
    # separately after prediction (see apply_policy). Keeps model and policy apart.
    # Excluded: "defaulted_in_next_12_months" (target leakage), "farmer_id"
    # (identifier), "application_year" (used only for the time split).

    X = df[features].copy()
    y = df["target_entitlement_score"]

    # print("Encoding categorical variables...")
    # encoder = LabelEncoder()
    # X["crop_type"] = encoder.fit_transform(X["crop_type"])
    # X["pm_kisan_status"] = encoder.fit_transform(X["pm_kisan_status"])

    # print("One-hot encoding categorical variables...")
    # One-hot avoids the fake ordering LabelEncoder imposes on categories
    # (e.g. Rice=2 > Cotton=0), which would mislead the model.
    # X = pd.get_dummies(X, columns=categorical_cols)
    # ^ now handled INSIDE the Pipeline below (OneHotEncoder), so X stays raw.

    # print("Splitting data...")
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, shuffle=True
    # )

    # Splitting data based on time (application_year) rather than random split (train on 2022 and 2023, test on 2024) for better scoring of future farmers
    print("Splitting data based on application_year")
    is_train = df["application_year"] < 2024
    X_train, X_test = X[is_train], X[~is_train]
    y_train, y_test = y[is_train], y[~is_train]

    # Handle missing values (rainfall_deviation_pct and ndvi_score are ~15% null).
    # Previously imputed by hand with the TRAIN median; now done INSIDE the
    # Pipeline (SimpleImputer) so it is fit on train only and saved with the model.
    # train_medians = X_train[numeric_cols].median()
    # X_train = X_train.fillna(train_medians)
    # X_test = X_test.fillna(train_medians)

    # print("Training XGBoost...")
    # model = xgb.XGBRegressor(
    #     n_estimators=60,
    #     max_depth=4,
    #     learning_rate=0.1,
    #     random_state=42,
    #     n_jobs=1,
    #     tree_method="hist",
    # )
    # model.fit(X_train, y_train)

    # All preprocessing now lives INSIDE the pipeline: fit on train only and
    # saved together with the model, so scoring new raw data is one call.
    #   - numeric: median imputation
    #   - categorical: most-frequent imputation + one-hot
    #     (handle_unknown="ignore" stays safe for unseen categories at scoring)
    print("Training XGBoost pipeline...")
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("xgb", xgb.XGBRegressor(
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        )),
    ])
    model.fit(X_train, y_train)

    # Model predicts the BASE score from genuine features only.
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"Validation R2 Score: {score:.4f} (Wow!)")

    # Policy is applied separately, on top of the base score.
    test_pm_kisan = df.loc[~is_train, "pm_kisan_status"]
    final_scores = apply_policy(preds, test_pm_kisan)
    print(f"PM-Kisan policy applied to {(test_pm_kisan == 'No').sum()} farmers "
          f"(-{PM_KISAN_PENALTY:.0f} each), separate from the model.")

    # ---- Save reproducible artifacts (for review / re-running) ----
    # Everything a reviewer needs to reproduce and audit this run: the full
    # pipeline, the feature contract, the validation result, and version info.
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model_pipeline.pkl")
    with open("artifacts/feature_list.json", "w") as f:
        json.dump({
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "policy_feature": "pm_kisan_status",
            "target": "target_entitlement_score",
        }, f, indent=2)
    with open("artifacts/validation_summary.json", "w") as f:
        json.dump({
            "r2": round(float(score), 4),
            "mae": round(float(mae), 2),
            "n_train": int(is_train.sum()),
            "n_test": int((~is_train).sum()),
            "split": "time-based: train year < 2024, test year == 2024",
        }, f, indent=2)
    with open("artifacts/metadata.json", "w") as f:
        json.dump({
            "xgboost_version": xgb.__version__,
            "random_state": 42,
            "pm_kisan_penalty": PM_KISAN_PENALTY,
        }, f, indent=2)
    print("Artifacts saved to ./artifacts/")


if __name__ == "__main__":
    train_model()
