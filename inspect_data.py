
# ==================================================
# OULAD PROJECT - DATA INSPECTION
# Person 3 - Exploratory Data Analysis
# ==================================================

import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Files to inspect
files = [
    "model_data.csv",
    "activity_features.csv"
]

print("=" * 60)
print("OULAD DATA INSPECTION")
print("=" * 60)

for filename in files:

    path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print("FILE:", filename)
    print("=" * 60)

    if not path.exists():
        print("WARNING: File not found!")
        continue

    try:
        # Load dataset
        df = pd.read_csv(path)

        # Remove spaces from column names
        df.columns = df.columns.str.strip()

        # Basic information
        print("\nDataset shape:")
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])

        # Column names
        print("\nColumn names:")
        for col in df.columns:
            print("-", col)

        # First five rows
        print("\nFirst 5 rows:")
        print(df.head().to_string())

        # Data types
        print("\nData types:")
        print(df.dtypes)

        # Missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Missing percentages
        print("\nMissing percentages:")
        print((df.isnull().mean() * 100).round(2))

        # Duplicate rows
        print("\nDuplicate rows:")
        print(df.duplicated().sum())

        # Unique values
        print("\nUnique values:")
        print(df.nunique())

        # Statistical summary
        print("\nStatistical summary:")
        print(df.describe(include="all").to_string())

        # Save inspection results
        with open(
            OUTPUT_DIR / f"{path.stem}_inspection.txt",
            "w",
            encoding="utf-8"
        ) as report:

            report.write(f"INSPECTION REPORT: {filename}\n")
            report.write("=" * 50 + "\n")

            report.write(f"\nShape: {df.shape}\n")
            report.write(f"\nColumns:\n{df.columns.tolist()}\n")
            report.write(f"\nData types:\n{df.dtypes}\n")
            report.write(f"\nMissing values:\n{df.isnull().sum()}\n")
            report.write(
                f"\nMissing percentages:\n"
                f"{(df.isnull().mean() * 100).round(2)}\n"
            )
            report.write(
                f"\nDuplicate rows: {df.duplicated().sum()}\n"
            )
            report.write(
                f"\nUnique values:\n{df.nunique()}\n"
            )
            report.write(
                f"\nStatistics:\n"
                f"{df.describe(include='all').to_string()}\n"
            )

        print("\nInspection report saved successfully!")

    except Exception as error:
        print("\nERROR:", error)

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("Reports saved in:", OUTPUT_DIR)
print("=" * 60)
