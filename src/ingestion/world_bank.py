import wbdata
import pandas as pd
import os
from datetime import datetime
import boto3
from io import StringIO

# dict that maps world bank codes to clear names
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",
    "DT.DOD.DECT.GN.ZS": "external_debt_gni",
    "FI.RES.TOTL.MO": "foreign_reserves_months",
    "NE.RSB.GNFS.ZS": "trade_balance_gdp",
    "SL.UEM.TOTL.ZS": "unemployment",
    "FR.INR.RINR": "real_interest_rate",
    "FM.LBL.BMNY.ZG": "broad_money_growth"
}

# list of 30 countries across emerging mrkts
# countries most historically prone to currency crisis
COUNTRIES = [
    "ARG", "BRA", "MEX", "CHL", "COL", "PER", "VEN",
    "THA", "IDN", "MYS", "PHL", "KOR", "TUR", "RUS",
    "ZAF", "NGA", "EGY", "GHA", "KEN", "MAR",
    "POL", "HUN", "CZE", "ROU", "BGR",
    "IND", "PAK", "BGD", "VNM", "UKR"
]

# time range
START_YEAR = 1990
END_YEAR = 2017 # stop here because that's where the Laeven & Valencia crisis labels end


# calls the World Bank API using the wbdata library
# pulls all 9 indicators for all 30 countries for every year from 1990 to 2017. 
# returns a clean df where each row is one country in one year.
def fetch_world_bank_data():
    print("Fetching World Bank indicators...")
    date_range = (datetime(START_YEAR, 1, 1), datetime(END_YEAR, 1, 1))
    df = wbdata.get_dataframe(
        INDICATORS,
        country=COUNTRIES,
        date=date_range)
    df = df.reset_index()
    df.columns = ["country", "year"] + list(INDICATORS.values())
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    print(f"Fetched {len(df)} rows for {df['country'].nunique()} countries.")
    return df

BUCKET_NAME = "currency-crisis-ews"

def save_to_s3(df, s3_key="raw/world_bank.csv"):
    print(f"Uploading to s3://{BUCKET_NAME}/{s3_key}...")
    s3 = boto3.client("s3")
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=csv_buffer.getvalue()
    )
    print(f"Saved to s3://{BUCKET_NAME}/{s3_key}")


if __name__ == "__main__":
    df = fetch_world_bank_data()
    save_to_s3(df)
    print(df.head())