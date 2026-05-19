"""
correlation_analysis.py

Analyzes feature correlations to identify and remove redundant features
before model training.

Steps:
1. Load processed features from S3
2. Compute correlation matrix
3. Identify highly correlated feature pairs (threshold > 0.85)
4. Remove redundant features keeping the most informative one
5. Save cleaned feature set back to S3

Output: s3://currency-crisis-ews/processed/features_clean.csv
"""

import pandas as pd
import boto3
from io import StringIO

BUCKET_NAME = "currency-crisis-ews"
CORRELATION_THRESHOLD = 0.85


#  1. Load data from S3 
def load_features():
    print("Loading features from S3...")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="processed/features.csv")
    df = pd.read_csv(obj["Body"])
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# 2. Compute correlations 
def compute_correlations(df):
    print("\nComputing correlations...")
    drop_cols = ["country", "year", "currency_crisis"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    corr_matrix = df[feature_cols].corr().abs()
    # find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > CORRELATION_THRESHOLD:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    round(corr_matrix.iloc[i, j], 3)
                ))

    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: -x[2])
    print(f"\n  Found {len(high_corr_pairs)} highly correlated pairs (>{CORRELATION_THRESHOLD}):")
    for a, b, c in high_corr_pairs:
        print(f"    {c}  |  {a}  vs  {b}")

    return corr_matrix, high_corr_pairs, feature_cols


# 3. Remove redundant features 
def remove_redundant_features(df, high_corr_pairs, feature_cols):
    print("\nRemoving redundant features...")
    # for each correlated pair, remove the second one
    # keeping the first which is usually the original indicator
    # rather than the lag/rolling version
    features_to_drop = set()
    for a, b, corr in high_corr_pairs:
        if b not in features_to_drop:
            features_to_drop.add(b)
            print(f"  Dropping: {b} (correlated with {a}, r={corr})")

    # keep non-feature columns + remaining features
    keep_cols = ["country", "year", "currency_crisis"] + [
        c for c in feature_cols if c not in features_to_drop]

    df_clean = df[keep_cols]
    print(f"\n  Original features: {len(feature_cols)}")
    print(f"  Dropped features:  {len(features_to_drop)}")
    print(f"  Remaining features: {len(keep_cols) - 3}")

    return df_clean


# 4. Save cleaned features to S3 
def save_to_s3(df, s3_key="processed/features_clean.csv"):
    print(f"\nUploading to s3://{BUCKET_NAME}/{s3_key}...")
    s3 = boto3.client("s3")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=csv_buffer.getvalue())
    print(f"  Saved to s3://{BUCKET_NAME}/{s3_key}")

if __name__ == "__main__":
    df = load_features()
    corr_matrix, high_corr_pairs, feature_cols = compute_correlations(df)
    df_clean = remove_redundant_features(df, high_corr_pairs, feature_cols)
    save_to_s3(df_clean)
    print(f"\nFinal clean dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print("\nRemaining features:")
    for col in df_clean.columns:
        if col not in ["country", "year", "currency_crisis"]:
            print(f"  - {col}")