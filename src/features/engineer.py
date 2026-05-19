"""
Merges World Bank, IMF WEO, and crisis label datasets from S3 into a single
clean panel dataset ready for model training.

Steps:
1. Load all three raw datasets from S3
2. Merge wb and imf on country + year
3. Merge with crisis labels
4. Handle missing values
5. Create lag features (t-1, t-2) for key indicators
6. Create rolling average features (3-year window)
7. Save final dataset back to S3

Output: s3://currency-crisis-ews/processed/features.csv
"""

import pandas as pd
import boto3
from io import StringIO

BUCKET_NAME = "currency-crisis-ews"


# 1. load data from s3 
def load_from_s3(s3_key):
    print(f"Loading s3://{BUCKET_NAME}/{s3_key}...")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    df = pd.read_csv(obj["Body"])
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# 2. merge datasets 
def merge_datasets(wb, imf, labels):
    print("Merging datasets...")
    # merge wb and imf on country + year
    df = pd.merge(wb, imf, on=["country", "year"], how="outer")
    # merge with crisis labels
    df = pd.merge(df, labels, on=["country", "year"], how="inner")
    print(f"  Merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# 3. Handle missing values 
def handle_missing(df):
    print("Handling missing values...")
    # fill missing values forward within each country
    # example: if 1995 data is missing, use 1994 value
    df = df.sort_values(["country", "year"])
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["year", "currency_crisis"]]
    df[numeric_cols] = df.groupby("country")[numeric_cols].transform(
        lambda x: x.fillna(method="ffill").fillna(method="bfill"))
    # drop rows that still have too many missing values (>50% of features)
    threshold = len(numeric_cols) * 0.5
    df = df.dropna(thresh=len(df.columns) - threshold)

    print(f"  After cleaning: {df.shape[0]} rows")
    return df


# 4. Create lag features 
def create_lag_features(df):
    print("Creating lag features...")
    # most predictive indicators for currency crises
    lag_cols = [
        "gdp_growth", "inflation", "current_account_gdp",
        "external_debt_gni", "foreign_reserves_months",
        "trade_balance_gdp", "real_interest_rate",
        "govt_debt_gdp", "broad_money_growth"]
    for col in lag_cols:
        if col in df.columns:
            # 1-year lag: value from previous year
            df[f"{col}_lag1"] = df.groupby("country")[col].shift(1)
            # 2-year lag: value from 2 years ago
            df[f"{col}_lag2"] = df.groupby("country")[col].shift(2)
    print(f"  Added lag features, so the dataset now has {df.shape[1]} columns")
    return df


# 5. Create rolling average features
def create_rolling_features(df):
    print("Creating rolling average features...")
    roll_cols = [
        "gdp_growth", "inflation", "current_account_gdp",
        "foreign_reserves_months", "broad_money_growth"]
    for col in roll_cols:
        if col in df.columns:
            # 3-year rolling avg (to smooth out short-term noise)
            df[f"{col}_roll3"] = df.groupby("country")[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean())

    print(f"  Added rolling features, so the dataset now has {df.shape[1]} columns")
    return df


# 6. Save to S3 
def save_to_s3(df, s3_key="processed/features.csv"):
    print(f"Uploading to s3://{BUCKET_NAME}/{s3_key}...")
    s3 = boto3.client("s3")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=csv_buffer.getvalue())
    print(f"Saved to s3://{BUCKET_NAME}/{s3_key}")


if __name__ == "__main__":
    # load
    wb = load_from_s3("raw/world_bank.csv")
    imf = load_from_s3("raw/imf_weo.csv")
    labels = load_from_s3("raw/crisis_labels.csv")
    # process
    df = merge_datasets(wb, imf, labels)
    df = handle_missing(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    # save
    save_to_s3(df)

    # summary
    print(f"\nFinal dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Crisis years: {df['currency_crisis'].sum()}")
    print(f"Non-crisis years: {(df['currency_crisis'] == 0).sum()}")
    print(df.head())