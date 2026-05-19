"""
AWS Lambda handler for the Currency Crisis Early Warning System API.

Accepts a POST request with a country name and year, loads the trained
Logistic Regression model from S3, and returns a crisis risk score
with the top contributing features.

Expected request body:
{
    "country": "Argentina",
    "year": 2015
}

Response:
{
    "country": "Argentina",
    "year": 2015,
    "risk_score": 0.73,
    "risk_level": "High",
    "top_features": [
        {"feature": "gdp_growth", "value": -2.3, "impact": "increases risk"},
        ...
    ]
}
"""

import json
import boto3
import joblib
import pandas as pd
import numpy as np

BUCKET_NAME = "currency-crisis-ews"
MODEL_KEY = "models/best_tuned_model.pkl"
FEATURES_KEY = "processed/features_clean.csv"
TMP_MODEL_PATH = "/tmp/best_tuned_model.pkl"

# risk level thresholds
RISK_THRESHOLDS = {
    "Low": (0.0, 0.3),
    "Medium": (0.3, 0.6),
    "High": (0.6, 1.0)}


# 1. Load model from S3 (cached after first call)
_model_bundle = None

def load_model():
    global _model_bundle
    if _model_bundle is None:
        print("Loading model from S3...")
        s3 = boto3.client("s3")
        s3.download_file(BUCKET_NAME, MODEL_KEY, TMP_MODEL_PATH)
        _model_bundle = joblib.load(TMP_MODEL_PATH)
    return _model_bundle["model"], _model_bundle["scaler"]


# 2. Load features from S3
def load_features():
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=FEATURES_KEY)
    return pd.read_csv(obj["Body"])


# 3. Get risk level from score
def get_risk_level(score):
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= score <= high:
            return level
    return "High"


# 4. Get top contributing features
def get_top_features(row, feature_cols, model, scaler, n=5):
    x = row[feature_cols].fillna(0).values.reshape(1, -1)
    x_scaled = scaler.transform(x)
    coefficients = model.coef_[0]
    contributions = x_scaled[0] * coefficients
    feature_impacts = sorted(
        zip(feature_cols, row[feature_cols].values, contributions),
        key=lambda x: abs(x[2]),
        reverse=True
    )[:n]
    return [
        {
            "feature": f,
            "value": round(float(v), 3) if not np.isnan(v) else None,
            "impact": "increases risk" if c > 0 else "decreases risk"
        }
        for f, v, c in feature_impacts]


# 5. Main Lambda handler
def handler(event, context):
    try:
        # parse request
        body = json.loads(event.get("body", "{}"))
        country = body.get("country")
        year = body.get("year")

        if not country or not year:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "country and year are required"})}

        # load model and data
        model, scaler = load_model()
        df = load_features()

        # find the row for the requested country and year
        drop_cols = ["country", "year", "currency_crisis"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        train_means = df[df["year"] <= 2007][feature_cols].mean()

        row = df[(df["country"] == country) & (df["year"] == year)]

        if row.empty:
            return {
                "statusCode": 404,
                "body": json.dumps({
                    "error": f"No data found for {country} in {year}"})}

        row = row.iloc[0]
        x = row[feature_cols].fillna(train_means).values.reshape(1, -1)
        x_scaled = scaler.transform(x)

        # get risk score
        risk_score = float(model.predict_proba(x_scaled)[0][1])
        risk_level = get_risk_level(risk_score)
        top_features = get_top_features(row, feature_cols, model, scaler)

        response = {
            "country": country,
            "year": year,
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level,
            "top_features": top_features}

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response)}

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})}