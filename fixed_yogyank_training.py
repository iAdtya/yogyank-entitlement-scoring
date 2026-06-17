"""
Yogyank Entitlement Score - Baseline Training Script (Draft v1)
Author: Junior Data Scientist
Notes: Model is performing well. Validation score looks good. Ready for production.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import xgboost as xgb
import joblib


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

    print("One-hot encoding categorical variables...")
    # One-hot avoids the fake ordering LabelEncoder imposes on categories
    # (e.g. Rice=2 > Cotton=0), which would mislead the model.
    X = pd.get_dummies(X, columns=categorical_cols)

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
    # Impute with the TRAIN median only, then apply to both, so no test-set
    # information leaks into training.
    train_medians = X_train[numeric_cols].median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    # Model predicts the BASE score from genuine features only.
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    print(f"Validation R2 Score: {score:.4f} (Wow!)")

    # Policy is applied separately, on top of the base score.
    test_pm_kisan = df.loc[~is_train, "pm_kisan_status"]
    final_scores = apply_policy(preds, test_pm_kisan)
    print(f"PM-Kisan policy applied to {(test_pm_kisan == 'No').sum()} farmers "
          f"(-{PM_KISAN_PENALTY:.0f} each), separate from the model.")

    joblib.dump(model, "xgboost_baseline.pkl")
    print("Model saved to xgboost_baseline.pkl")


if __name__ == "__main__":
    train_model()
