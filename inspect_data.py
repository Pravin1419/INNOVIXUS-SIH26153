import pandas as pd

FILE = r".\data\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"

print("Inspecting dataset structure...")
print("Please wait...\n")

# Read only a few rows
df = pd.read_csv(
    FILE,
    nrows=5,
    low_memory=False
)

print("=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)

print(f"Number of columns : {len(df.columns)}")

print("\nCOLUMN NAMES")
print("-" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\n" + "=" * 60)
print("SAMPLE DATA")
print("=" * 60)

print(df.head(2).to_string())

print("\nInspection complete!")