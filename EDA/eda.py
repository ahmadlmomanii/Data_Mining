
# ============================================================
# OULAD PROJECT - EXPLORATORY DATA ANALYSIS (EDA)

# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = DATA_DIR / "model_data.csv"
ACTIVITY_FILE = DATA_DIR / "activity_features.csv"

sns.set_theme(style="whitegrid")

print("=" * 60)
print("OULAD EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(MODEL_FILE)
activity = pd.read_csv(ACTIVITY_FILE)

print("\nModel data shape:", df.shape)
print("Activity data shape:", activity.shape)

# ------------------------------------------------------------
# 3. DATA QUALITY ANALYSIS
# ------------------------------------------------------------

quality_report = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Missing Percentage": (df.isnull().mean() * 100).round(2),
    "Unique Values": df.nunique()
})

quality_report.to_csv(
    REPORTS_DIR / "data_quality_report.csv"
)

print("\nDATA QUALITY REPORT")
print(quality_report)

print("\nDuplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

numeric_stats = df.describe()

numeric_stats.to_csv(
    REPORTS_DIR / "descriptive_statistics.csv"
)

print("\nDESCRIPTIVE STATISTICS")
print(numeric_stats)

# ------------------------------------------------------------
# 5. FINAL RESULT DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="final_result",
    order=df["final_result"].value_counts().index
)

plt.title("Distribution of Students' Final Results")
plt.xlabel("Final Result")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig(CHARTS_DIR / "final_result_distribution.png")
plt.close()

print("\nFinal result chart saved.")

# ------------------------------------------------------------
# 6. DEMOGRAPHIC ANALYSIS
# ------------------------------------------------------------

demographic_columns = [
    "gender",
    "region",
    "highest_education",
    "age_band",
    "disability"
]

for column in demographic_columns:

    if column not in df.columns:
        continue

    plt.figure(figsize=(10, 5))

    order = df[column].value_counts().index

    sns.countplot(
        data=df,
        x=column,
        order=order
    )

    plt.title(f"Student Distribution by {column}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Number of Students")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(CHARTS_DIR / f"{column}_distribution.png")
    plt.close()

# ------------------------------------------------------------
# 7. DEMOGRAPHICS VS FINAL RESULTS
# ------------------------------------------------------------

for column in demographic_columns:

    if column not in df.columns:
        continue

    result_table = pd.crosstab(
        df[column],
        df["final_result"],
        normalize="index"
    ) * 100

    result_table.to_csv(
        REPORTS_DIR / f"{column}_vs_final_result.csv"
    )

    plt.figure(figsize=(10, 6))

    result_table.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6)
    )

    plt.title(f"{column} vs Final Result")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Final Result")
    plt.tight_layout()

    plt.savefig(CHARTS_DIR / f"{column}_vs_final_result.png")
    plt.close()

# ------------------------------------------------------------
# 8. NUMERIC FEATURES DISTRIBUTION
# ------------------------------------------------------------

numeric_columns = [
    "num_of_prev_attempts",
    "studied_credits",
    "total_clicks_first_28_days",
    "active_days_first_28_days",
    "unique_sites_first_28_days"
]

for column in numeric_columns:

    if column not in df.columns:
        continue

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(CHARTS_DIR / f"{column}_distribution.png")
    plt.close()

# ------------------------------------------------------------
# 9. ACTIVITY FEATURES VS FINAL RESULTS
# ------------------------------------------------------------

activity_columns = [
    "total_clicks_first_28_days",
    "active_days_first_28_days",
    "unique_sites_first_28_days"
]

for column in activity_columns:

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="final_result",
        y=column
    )

    plt.title(f"{column} vs Final Result")
    plt.xlabel("Final Result")
    plt.ylabel(column.replace("_", " ").title())
    plt.tight_layout()

    plt.savefig(CHARTS_DIR / f"{column}_vs_final_result.png")
    plt.close()

# ------------------------------------------------------------
# 10. CORRELATION ANALYSIS
# ------------------------------------------------------------

correlation_columns = [
    "num_of_prev_attempts",
    "studied_credits",
    "total_clicks_first_28_days",
    "active_days_first_28_days",
    "unique_sites_first_28_days"
]

correlation_data = df[correlation_columns].corr()

correlation_data.to_csv(
    REPORTS_DIR / "correlation_matrix.csv"
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_data,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numeric Features")
plt.tight_layout()

plt.savefig(CHARTS_DIR / "correlation_heatmap.png")
plt.close()

# ------------------------------------------------------------
# 11. ACTIVITY SUMMARY
# ------------------------------------------------------------

activity_summary = df[
    activity_columns
].groupby(df["final_result"]).agg(
    ["mean", "median", "std"]
).round(2)

activity_summary.to_csv(
    REPORTS_DIR / "activity_vs_results_summary.csv"
)

print("\nACTIVITY VS FINAL RESULTS")
print(activity_summary)

# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

summary = f"""
OULAD EDA SUMMARY

Number of students: {len(df)}

Number of columns: {len(df.columns)}

Missing values: {df.isnull().sum().sum()}

Duplicate rows: {df.duplicated().sum()}

Final result distribution:
{df['final_result'].value_counts().to_string()}

Average total clicks:
{df['total_clicks_first_28_days'].mean():.2f}

Average active days:
{df['active_days_first_28_days'].mean():.2f}

Average unique sites:
{df['unique_sites_first_28_days'].mean():.2f}
"""

with open(
    REPORTS_DIR / "eda_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

print("\n" + summary)

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("Charts saved in:", CHARTS_DIR)
print("Reports saved in:", REPORTS_DIR)
print("=" * 60)
