import pandas as pd
from pathlib import Path

# Path to raw data
data_path = Path("data/raw")

print("Loading studentVle.csv...")

# Read only the columns we need
student_vle = pd.read_csv(
    data_path / "studentVle.csv",
    usecols=[
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click"
    ]
)

print("Data loaded successfully.")
print("Original shape:", student_vle.shape)

# Keep activity from the first 28 days
first_28_days = student_vle[
    (student_vle["date"] >= 0) &
    (student_vle["date"] <= 27)
].copy()

print("\nFirst 28 days shape:", first_28_days.shape)

# Create activity features for each student
activity_features = (
    first_28_days
    .groupby(
        ["code_module", "code_presentation", "id_student"]
    )
    .agg(
        total_clicks_first_28_days=("sum_click", "sum"),
        active_days_first_28_days=("date", "nunique"),
        unique_sites_first_28_days=("id_site", "nunique")
    )
    .reset_index()
)

print("\nActivity features created.")
print("Feature dataset shape:", activity_features.shape)

print("\nFirst 5 rows:")
print(activity_features.head())

print("\nSummary:")
print(activity_features.describe())

# Save the features
output_path = data_path.parent / "activity_features.csv"

activity_features.to_csv(
    output_path,
    index=False
)

print("\nSaved successfully to:")
print(output_path)