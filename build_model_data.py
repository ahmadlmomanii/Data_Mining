import pandas as pd
from pathlib import Path

# Paths
data_path = Path("data/raw")
features_path = Path("data/activity_features.csv")

print("Loading student information...")

# Load student information
student_info = pd.read_csv(
    data_path / "studentInfo.csv"
)

print("Student info shape:", student_info.shape)

print("\nLoading activity features...")

# Load early activity features
activity_features = pd.read_csv(
    features_path
)

print("Activity features shape:", activity_features.shape)

# Merge student information with early activity
model_data = student_info.merge(
    activity_features,
    on=[
        "code_module",
        "code_presentation",
        "id_student"
    ],
    how="left"
)

print("\nAfter merge:")
print("Model data shape:", model_data.shape)

# Fill missing activity with 0
activity_columns = [
    "total_clicks_first_28_days",
    "active_days_first_28_days",
    "unique_sites_first_28_days"
]

model_data[activity_columns] = (
    model_data[activity_columns]
    .fillna(0)
)

# Check final result distribution
print("\nFinal Result Distribution:")
print(model_data["final_result"].value_counts())

print("\nFinal Result Percentages:")
print(
    model_data["final_result"]
    .value_counts(normalize=True) * 100
)

# Check missing values
print("\nMissing values:")
print(model_data.isnull().sum())

# Save preliminary dataset
output_path = Path("data/model_data.csv")

model_data.to_csv(
    output_path,
    index=False
)

print("\nModel data saved successfully:")
print(output_path)

print("\nFinal dataset shape:")
print(model_data.shape)

print("\nFirst 5 rows:")
print(model_data.head())