"""
Hyperparameter tuning for Logistic Regression and XGBoost models.

Steps:
1. Load clean features from S3
2. Time-based train/test split (train: 1990-2007, test: 2008-2017)
3. Apply SMOTE
4. GridSearchCV for Logistic Regression
5. RandomizedSearchCV for XGBoost
6. Compare tuned models
7. Save best tuned model to S3

Output: s3://currency-crisis-ews/models/best_tuned_model.pkl
"""

import pandas as pd
import numpy as np
import boto3
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BUCKET_NAME = "currency-crisis-ews"
TRAIN_END_YEAR = 2007
TEST_START_YEAR = 2008


# 1. Load data from S3
def load_features():
    print("Loading clean features from S3...")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="processed/features_clean.csv")
    df = pd.read_csv(obj["Body"])
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# 2. Time-based train/test split
def split_data(df):
    print(f"Splitting data — train: 1990-{TRAIN_END_YEAR}, test: {TEST_START_YEAR}-2017...")
    train = df[df["year"] <= TRAIN_END_YEAR]
    test = df[df["year"] >= TEST_START_YEAR]
    drop_cols = ["country", "year", "currency_crisis"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    train_means = train[feature_cols].mean()
    X_train = train[feature_cols].fillna(train_means)
    y_train = train["currency_crisis"]
    X_test = test[feature_cols].fillna(train_means)
    y_test = test["currency_crisis"]
    print(f"  Train: {X_train.shape[0]} rows, {y_train.sum()} crisis years")
    print(f"  Test:  {X_test.shape[0]} rows, {y_test.sum()} crisis years")
    return X_train, X_test, y_train, y_test, feature_cols


# 3. Apply SMOTE
def apply_smote(X_train, y_train):
    print("Applying SMOTE...")
    X_train = X_train.fillna(X_train.mean())
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE — crisis: {y_resampled.sum()}, non-crisis: {(y_resampled == 0).sum()}")
    return X_resampled, y_resampled


# 4. GridSearchCV for Logistic Regression
def tune_logistic(X_train, y_train, X_test, y_test):
    print("\nTuning Logistic Regression with GridSearchCV...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "l1_ratio": [0, 1],
    "solver": ["saga"],
    "penalty": ["elasticnet"]}
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1)

    grid_search.fit(X_train_scaled, y_train)
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV AUC: {grid_search.best_score_:.3f}")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    results = evaluate_model("Logistic Regression (tuned)", y_test, y_pred, y_prob)
    return best_model, scaler, results


# 5. RandomizedSearchCV for XGBoost
def tune_xgboost(X_train, y_train, X_test, y_test):
    print("\nTuning XGBoost with RandomizedSearchCV...")
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "scale_pos_weight": [5, 10, 15, 20]}

    random_search = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric="auc", verbosity=0),
        param_distributions=param_dist,
        n_iter=50,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        random_state=42,
        verbose=1)

    random_search.fit(X_train, y_train)
    print(f"  Best params: {random_search.best_params_}")
    print(f"  Best CV AUC: {random_search.best_score_:.3f}")

    best_model = random_search.best_estimator_
    X_test_clean = X_test.fillna(X_test.mean())
    y_pred = best_model.predict(X_test_clean)
    y_prob = best_model.predict_proba(X_test_clean)[:, 1]
    results = evaluate_model("XGBoost (tuned)", y_test, y_pred, y_prob)
    return best_model, results


# 6. Evaluate model
def evaluate_model(name, y_test, y_pred, y_prob):
    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Precision: {precision:.3f}")
    return {"model": name, "auc": auc, "recall": recall, "precision": precision}


# 7. Print comparison table
def print_comparison(all_results):
    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'AUC':>8} {'Recall':>8} {'Precision':>10}")
    print("=" * 55)
    for r in all_results:
        print(f"{r['model']:<25} {r['auc']:>8.3f} {r['recall']:>8.3f} {r['precision']:>10.3f}")
    print("=" * 55)


# 8. Save best tuned model to S3
def save_model_to_s3(model, scaler=None):
    print(f"\nSaving best tuned model to s3://{BUCKET_NAME}/models/best_tuned_model.pkl...")
    joblib.dump({"model": model, "scaler": scaler}, "/tmp/best_tuned_model.pkl")
    s3 = boto3.client("s3")
    s3.upload_file("/tmp/best_tuned_model.pkl", BUCKET_NAME, "models/best_tuned_model.pkl")
    print(f"  Saved to s3://{BUCKET_NAME}/models/best_tuned_model.pkl")


if __name__ == "__main__":
    df = load_features()
    X_train, X_test, y_train, y_test, feature_cols = split_data(df)
    X_train, y_train = apply_smote(X_train, y_train)

    all_results = []

    lr_model, scaler, lr_results = tune_logistic(X_train, y_train, X_test, y_test)
    all_results.append(lr_results)

    xgb_model, xgb_results = tune_xgboost(X_train, y_train, X_test, y_test)
    all_results.append(xgb_results)

    print_comparison(all_results)

    # save best model by AUC
    best = max(all_results, key=lambda x: x["auc"])
    print(f"\nBest tuned model: {best['model']} (AUC: {best['auc']:.3f})")

    if "Logistic" in best["model"]:
        save_model_to_s3(lr_model, scaler)
    else:
        save_model_to_s3(xgb_model)

    print("\nDone!")