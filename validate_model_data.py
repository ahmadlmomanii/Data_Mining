import pandas as pd
from pathlib import Path

# Path to model data
data_path = Path("data/model_data.csv")

print("Loading model_data.csv...")

df = pd.read_csv(data_path)

print("\nDataset shape:")
print(df.shape)

# --------------------------------------------------
# 1. Check duplicate rows
# --------------------------------------------------

print("\n1. Duplicate rows:")
print(df.duplicated().sum())

# --------------------------------------------------
# 2. Check duplicate students
# --------------------------------------------------

print("\n2. Duplicate student records:")

student_duplicates = df.duplicated(
    subset=[
        "code_module",
        "code_presentation",
        "id_student"
    ]
).sum()

print(student_duplicates)

# --------------------------------------------------
# 3. Check missing values
# --------------------------------------------------

print("\n3. Missing values:")
print(df.isnull().sum())

# --------------------------------------------------
# 4. Check activity feature values
# --------------------------------------------------

activity_columns = [
    "total_clicks_first_28_days",
    "active_days_first_28_days",
    "unique_sites_first_28_days"
]

print("\n4. Activity feature summary:")
print(df[activity_columns].describe())

# --------------------------------------------------
# 5. Check negative values
# --------------------------------------------------

print("\n5. Negative activity values:")

for column in activity_columns:
    negative_count = (df[column] < 0).sum()
    print(f"{column}: {negative_count}")

# --------------------------------------------------
# 6. Students with zero activity
# --------------------------------------------------

zero_activity = (
    (df["total_clicks_first_28_days"] == 0) &
    (df["active_days_first_28_days"] == 0) &
    (df["unique_sites_first_28_days"] == 0)
).sum()

print("\n6. Students with zero activity in first 28 days:")
print(zero_activity)

# --------------------------------------------------
# 7. Final result distribution
# --------------------------------------------------

print("\n7. Final result distribution:")
print(df["final_result"].value_counts())

# --------------------------------------------------
# 8. Check final result percentages
# --------------------------------------------------

print("\n8. Final result percentages:")
print(
    df["final_result"].value_counts(normalize=True) * 100
)

print("\nValidation completed successfully.")