import pandas as pd
import boto3
from io import StringIO


"""
build_labels.py

Builds the target variable (currency_crisis) for the early warning system.

Currency crisis episodes are hardcoded based on:
- Laeven & Valencia (2020): Systemic Banking Crises Database II
- Frankel & Rose (1996): Currency Crashes in Emerging Markets

A crisis year is marked as 1, all other years as 0.
The output is a panel dataset with one row per country per year (1990-2017)
covering 30 emerging market economies.

Output: s3://currency-crisis-ews/raw/crisis_labels.csv
"""

BUCKET_NAME = "currency-crisis-ews"

# documented currency crises from Laeven & Valencia (2020)
# and Frankel & Rose (1996): country, start year, end year
CURRENCY_CRISES = [
    ("Argentina", 1995, 1995),
    ("Argentina", 2001, 2002),
    ("Brazil", 1999, 1999),
    ("Brazil", 2002, 2002),
    ("Mexico", 1994, 1995),
    ("Colombia", 1998, 1999),
    ("Peru", 1990, 1990),
    ("Venezuela", 1994, 1995),
    ("Venezuela", 2002, 2003),
    ("Chile", 1998, 1999),
    ("Thailand", 1997, 1998),
    ("Indonesia", 1997, 1998),
    ("Malaysia", 1997, 1998),
    ("Philippines", 1997, 1998),
    ("Korea", 1997, 1998),
    ("Turkey", 1994, 1994),
    ("Turkey", 2001, 2001),
    ("Russia", 1998, 1999),
    ("Ukraine", 1998, 1999),
    ("Ukraine", 2008, 2009),
    ("South Africa", 1996, 1996),
    ("South Africa", 2001, 2002),
    ("Nigeria", 1999, 1999),
    ("Ghana", 1999, 2000),
    ("Kenya", 1993, 1993),
    ("Morocco", 1990, 1990),
    ("Egypt", 2003, 2003),
    ("India", 1991, 1991),
    ("Pakistan", 1996, 1996),
    ("Bangladesh", 1990, 1990),
    ("Hungary", 2008, 2009),
    ("Romania", 1996, 1997),
    ("Bulgaria", 1996, 1997),
    ("Poland", 1992, 1992),
    ("Vietnam", 1997, 1998),]

COUNTRIES = [
    "Argentina", "Brazil", "Mexico", "Chile", "Colombia", "Peru", "Venezuela",
    "Thailand", "Indonesia", "Malaysia", "Philippines", "Korea", "Turkey", "Russia",
    "South Africa", "Nigeria", "Egypt", "Ghana", "Kenya", "Morocco",
    "Poland", "Hungary", "Czech Republic", "Romania", "Bulgaria",
    "India", "Pakistan", "Bangladesh", "Vietnam", "Ukraine"]

START_YEAR = 1990
END_YEAR = 2017


def build_labels():
    print("Building crisis labels...")
    # create full panel — every country every year
    rows = []
    for country in COUNTRIES:
        for year in range(START_YEAR, END_YEAR + 1):
            rows.append({"country": country, "year": year, "currency_crisis": 0})
    df = pd.DataFrame(rows)
    # mark crisis years as 1
    for country, start, end in CURRENCY_CRISES:
        for year in range(start, end + 1):
            mask = (df["country"] == country) & (df["year"] == year)
            df.loc[mask, "currency_crisis"] = 1

    total_crises = df["currency_crisis"].sum()
    print(f"Built {len(df)} rows — {total_crises} crisis-years across {df['country'].nunique()} countries.")
    return df


def save_to_s3(df, s3_key="raw/crisis_labels.csv"):
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
    df = build_labels()
    save_to_s3(df)
    print(df[df["currency_crisis"] == 1].head(20))