import pandas as pd
from pathlib import Path

data_path = Path("data/raw")

files = [
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "studentVle.csv",
    "assessments.csv",
    "courses.csv",
    "vle.csv"
]

for file in files:
    df = pd.read_csv(data_path / file)

    print("\n" + "=" * 60)
    print(f"FILE: {file}")
    print("=" * 60)

    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicates:", df.duplicated().sum())
    print("\n" + "=" * 60)
print("FINAL RESULT DISTRIBUTION")
print("=" * 60)

student_info = pd.read_csv(data_path / "studentInfo.csv")

print(student_info["final_result"].value_counts())

print("\nPercentages:")
print(student_info["final_result"].value_counts(normalize=True) * 100)
student_vle = pd.read_csv(data_path / "studentVle.csv")

print("\n" + "=" * 60)
print("STUDENT VLE DUPLICATE CHECK")
print("=" * 60)

duplicate_rows = student_vle[
    student_vle.duplicated(
        subset=[
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date"
        ],
        keep=False
    )
]

print("Number of duplicated rows based on key columns:",
      len(duplicate_rows))

print("\nExample duplicated records:")
print(duplicate_rows.head(10))