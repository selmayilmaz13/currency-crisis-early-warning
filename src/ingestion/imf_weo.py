import pandas as pd
import boto3
from io import StringIO

BUCKET_NAME = "currency-crisis-ews"

# same 30 countries as the world bank script
COUNTRIES = [
    "Argentina", "Brazil", "Mexico", "Chile", "Colombia", "Peru", "Venezuela",
    "Thailand", "Indonesia", "Malaysia", "Philippines", "Korea", "Turkey", "Russia",
    "South Africa", "Nigeria", "Egypt", "Ghana", "Kenya", "Morocco",
    "Poland", "Hungary", "Czech Republic", "Romania", "Bulgaria",
    "India", "Pakistan", "Bangladesh", "Vietnam", "Ukraine"]

INDICATORS = {
    "BCA_NGDPD": "current_account_gdp_imf",
    "GGR_NGDP": "govt_revenue_gdp",
    "GGX_NGDP": "govt_expenditure_gdp",
    "GGXWDG_NGDP": "govt_debt_gdp",
    "NGDP_RPCH": "gdp_growth_imf",
    "PCPIPCH": "inflation_imf",
    "LUR": "unemployment_imf"}

START_YEAR = 1990
END_YEAR = 2017

# reads the raw excel file downloaded from the IMF website
def load_weo_data(filepath="data/raw/WEOOct2023all.xls"):
    print("Loading IMF WEO data...")
    df = pd.read_csv(filepath, sep="\t", encoding="latin-1", low_memory=False)
    print(f"Shape: {df.shape}")
    return df

# filters to only keep our 30 countries and 7 indicators out of the full dataset and 1990-2017 timeframe
# pivot to long format
# pivots it back so each indicator becomes its own column
def process_weo_data(df):
    print("Processing IMF WEO data...")
    df = df[df["Country"].isin(COUNTRIES)]
    df = df[df["WEO Subject Code"].isin(INDICATORS.keys())]
    year_cols = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
    keep_cols = ["Country", "WEO Subject Code"] + year_cols
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols]
    df = df.melt(
        id_vars=["Country", "WEO Subject Code"],
        var_name="year",
        value_name="value")
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(
        df["value"].astype(str).str.replace(",", ""),
        errors="coerce")
    df = df.pivot_table(
        index=["Country", "year"],
        columns="WEO Subject Code",
        values="value").reset_index()
    df.columns.name = None
    df = df.rename(columns={"Country": "country", **INDICATORS})
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    print(f"Processed {len(df)} rows for {df['country'].nunique()} countries.")
    return df

#converts the df to a CSV string in memory using 
# StringIO and uploads it directly to your S3 bucket
def save_to_s3(df, s3_key="raw/imf_weo.csv"):
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
    df_raw = load_weo_data()
    df = process_weo_data(df_raw)
    save_to_s3(df)
    print(df.head())