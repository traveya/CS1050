# Give a report on k-anonymity
import pandas as pd

df = pd.read_csv("raw_data.csv", dtype=str, keep_default_na=False)

# Get key information
pos_cols = ["cc_by_ip", "city", "postalCode", "LoE", "YoB", "gender"]

data_cols = [
    column for column in pos_cols
    if column in df.columns
]

excluded_cols = [
    column for column in pos_cols
    if column not in df.columns
]

print("Analyzing:", data_cols)
print("Absent columns:", excluded_cols)

if not data_cols:
    raise ValueError("no quasi-identifiers found! k irrelevant")

# Get data information
num_rows = len(df)

if num_rows == 0:
    raise ValueError("no file found") 

# Group rows that match on ALL remaining quasi-identifiers.
# dropna=False includes rows with missing values.
groups = (
    df.groupby(data_cols, dropna=False)
      .size()
      .reset_index(name="group_size")
)

sizes = groups["group_size"]
k = int(sizes.min())
unique_rows = int((sizes == 1).sum())

print("\nK-ANON REPORT")
print(f"Total rows:      {num_rows:,}")
print(f"Distinct groups: {len(groups):,}")
print(f"k-value:       {k}")
print(f"Unique rows:     {unique_rows:,} ({unique_rows / num_rows:.2%})")

# Count rows requiring suppression for each target.
print("\nSUPPRESSION NEEDED:")

for target_k in [2, 5, 10, 20]:
    suppressed = int(sizes[sizes < target_k].sum())
    retained = num_rows - suppressed

    print(
        f"k={target_k}: suppress {suppressed:,} rows "
        f"({suppressed / num_rows:.2%}), "
        f"retain {retained:,}"
    )

print("\nEND K-ANON REPORT")