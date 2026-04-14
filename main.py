import pandas as pd
import numpy as np

file_path = "medical_students_dataset.csv"
df_initial = pd.read_csv(file_path)
df = df_initial.copy()

def fill_missing_values(df):
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

df_cleaned = fill_missing_values(df.copy())

df_cleaned["Diabetes"] = df_cleaned["Diabetes"].map({"Yes": 1, "No": 0})
df_cleaned["Smoking"] = df_cleaned["Smoking"].map({"Yes": 1, "No": 0})
df_cleaned["Gender"] = df_cleaned["Gender"].map({"Female": 0, "Male": 1})

df_cleaned["Age"] = pd.to_numeric(df_cleaned["Age"], errors="coerce")
df_cleaned["Age"] = df_cleaned["Age"].replace([np.inf, -np.inf], np.nan)
df_cleaned["Age"] = df_cleaned["Age"].fillna(df_cleaned["Age"].mean())
df_cleaned["Age"] = df_cleaned["Age"].round(0).astype(int)

df_cleaned.to_csv("medical_students_dataset_cleaned.csv", index=False)

print("Dataset curatat a fost salvat in medical_students_dataset_cleaned.csv")
print(df_cleaned.head())
print("\nValori lipsa ramase:")
print(df_cleaned.isna().sum())