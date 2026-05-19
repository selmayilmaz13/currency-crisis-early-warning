"""
Generates SHAP explainability plots for the best trained model
(Logistic Regression) to interpret which features drive currency
crisis predictions.

Steps:
1. Load clean features from S3
2. Load best tuned model from S3
3. Compute SHAP values
4. Generate and save plots:
   - Summary plot (feature importance overview)
   - Bar plot (mean absolute SHAP values)
   - Waterfall plot (single prediction explanation)
5. Upload all plots to S3

Output: s3://currency-crisis-ews/outputs/shap_*.png
"""

import pandas as pd
import numpy as np
import boto3
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from io import BytesIO

BUCKET_NAME = "currency-crisis-ews"
TRAIN_END_YEAR = 2007

# 1. Load features from S3
def load_features():
    print("Loading clean features from S3...")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="processed/features_clean.csv")
    df = pd.read_csv(obj["Body"])
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# 2. Load model from S3
def load_model():
    print("Loading best tuned model from S3...")
    s3 = boto3.client("s3")
    s3.download_file(BUCKET_NAME, "models/best_tuned_model.pkl", "/tmp/best_tuned_model.pkl")
    bundle = joblib.load("/tmp/best_tuned_model.pkl")
    model = bundle["model"]
    scaler = bundle["scaler"]
    print("  Model loaded successfully")
    return model, scaler


# 3. Prepare data
def prepare_data(df, scaler):
    print("Preparing data...")
    train = df[df["year"] <= TRAIN_END_YEAR]
    drop_cols = ["country", "year", "currency_crisis"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    train_means = train[feature_cols].mean()
    X = df[feature_cols].fillna(train_means)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    print(f"  Prepared {X_scaled_df.shape[0]} rows")
    return X_scaled_df, feature_cols


# 4. Compute SHAP values
def compute_shap(model, X):
    print("Computing SHAP values...")
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    print("  SHAP values computed")
    return explainer, shap_values


# 5. Save plot to S3
def save_plot_to_s3(fig, filename):
    s3 = boto3.client("s3")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    s3_key = f"outputs/{filename}"
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=buf.getvalue())
    print(f"  Saved to s3://{BUCKET_NAME}/{s3_key}")
    buf.close()


# 6. Generate summary plot
def plot_summary(shap_values, X, feature_cols):
    print("Generating summary plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_cols,
        show=False)
    plt.title("SHAP Summary Plot — Currency Crisis Early Warning", fontsize=14)
    plt.tight_layout()
    save_plot_to_s3(plt.gcf(), "shap_summary.png")
    plt.close()


# 7. Generate bar plot
def plot_bar(shap_values, X, feature_cols):
    print("Generating bar plot...")
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_cols,
        plot_type="bar",
        show=False)
    plt.title("Mean Absolute SHAP Values — Feature Importance", fontsize=14)
    plt.tight_layout()
    save_plot_to_s3(plt.gcf(), "shap_bar.png")
    plt.close()


# 8. Generate waterfall plot for a single crisis prediction
def plot_waterfall(explainer, shap_values, X, df, feature_cols):
    print("Generating waterfall plot...")
    # find a crisis year to explain
    crisis_mask = df["currency_crisis"] == 1
    crisis_idx = df[crisis_mask].index[0] if crisis_mask.any() else 0

    explanation = shap.Explanation(
        values=shap_values[crisis_idx],
        base_values=explainer.expected_value,
        data=X.iloc[crisis_idx].values,
        feature_names=feature_cols)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(explanation, show=False)
    country = df.iloc[crisis_idx]["country"]
    year = df.iloc[crisis_idx]["year"]
    plt.title(f"SHAP Waterfall — {country} {year}", fontsize=14)
    plt.tight_layout()
    save_plot_to_s3(plt.gcf(), "shap_waterfall.png")
    plt.close()


if __name__ == "__main__":
    df = load_features()
    model, scaler = load_model()
    X_scaled, feature_cols = prepare_data(df, scaler)
    explainer, shap_values = compute_shap(model, X_scaled)
    plot_summary(shap_values, X_scaled, feature_cols)
    plot_bar(shap_values, X_scaled, feature_cols)
    plot_waterfall(explainer, shap_values, X_scaled, df, feature_cols)
    print("\nDone! All SHAP plots saved to S3.")