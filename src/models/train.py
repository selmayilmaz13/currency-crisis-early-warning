"""
Trains multiple currency crisis early warning models and compares performance.

Steps:
1. Load clean features from S3
2. Time-based train/test split (train: 1990-2007, test: 2008-2017)
3. Handle class imbalance with SMOTE
4. Train Logistic Regression baseline
5. Train Random Forest
6. Train LightGBM
7. Train XGBoost
8. Compare all models and save the best one to S3

Output: s3://currency-crisis-ews/models/xgboost_model.json
"""

import pandas as pd
import numpy as np
import boto3
import joblib
from io import StringIO, BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import lightgbm as lgb

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


# 3. Handle class imbalance with SMOTE
def apply_smote(X_train, y_train):
    print("Applying SMOTE...")
    X_train = X_train.fillna(X_train.mean())
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE — crisis: {y_resampled.sum()}, non-crisis: {(y_resampled == 0).sum()}")
    return X_resampled, y_resampled

# 4. Train Logistic Regression baseline 
def train_logistic(X_train, y_train, X_test, y_test):
    print("\nTraining Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    results = evaluate_model("Logistic Regression", y_test, y_pred, y_prob)
    return model, scaler, results

# 5. Train Random Forest
def train_random_forest(X_train, y_train, X_test, y_test):
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        class_weight="balanced",
        random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = evaluate_model("Random Forest", y_test, y_pred, y_prob)
    return model, results

# 6. Train LightGBM
def train_lightgbm(X_train, y_train, X_test, y_test):
    print("\nTraining LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=10,
        random_state=42,
        verbosity=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = evaluate_model("LightGBM", y_test, y_pred, y_prob)
    return model, results

# 7. Train XGBoost 
def train_xgboost(X_train, y_train, X_test, y_test):
    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        random_state=42,
        eval_metric="auc",
        verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = evaluate_model("XGBoost", y_test, y_pred, y_prob)
    return model, results

# 8. Evaluate model
def evaluate_model(name, y_test, y_pred, y_prob):
    auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Precision: {precision:.3f}")
    return {"model": name, "auc": auc, "recall": recall, "precision": precision}

# 9. Print comparison table
def print_comparison(all_results):
    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'AUC':>8} {'Recall':>8} {'Precision':>10}")
    print("=" * 55)
    for r in all_results:
        print(f"{r['model']:<25} {r['auc']:>8.3f} {r['recall']:>8.3f} {r['precision']:>10.3f}")
    print("=" * 55)

# 10. Save best model to S3
def save_model_to_s3(model, scaler, s3_key="models/best_model.pkl"):
    print(f"\nSaving best model to s3://{BUCKET_NAME}/{s3_key}...")
    joblib.dump({"model": model, "scaler": scaler}, "/tmp/best_model.pkl")
    s3 = boto3.client("s3")
    s3.upload_file("/tmp/best_model.pkl", BUCKET_NAME, "models/best_model.pkl")
    print(f"  Saved to s3://{BUCKET_NAME}/models/best_model.pkl")


if __name__ == "__main__":
    df = load_features()
    X_train, X_test, y_train, y_test, feature_cols = split_data(df)
    X_train, y_train = apply_smote(X_train, y_train)

    all_results = []

    lr_model, scaler, lr_results = train_logistic(X_train, y_train, X_test, y_test)
    all_results.append(lr_results)

    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    all_results.append(rf_results)

    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_test, y_test)
    all_results.append(lgb_results)

    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    all_results.append(xgb_results)

    print_comparison(all_results)

    # save best model by AUC
    best = max(all_results, key=lambda x: x["auc"])
    print(f"\nBest model: {best['model']} (AUC: {best['auc']:.3f})")
    save_model_to_s3(lr_model, scaler)

    print("\nDone!")