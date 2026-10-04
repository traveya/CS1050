# Give a report on k-anonymity
import pandas as pd

df = pd.read_csv("anonymized_data.csv", dtype=str, keep_default_na=False)

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

# Suppress rows in groups smaller than k=5
target_k = 5

row_group_sizes = (
    df.groupby(data_cols, dropna=False)[data_cols[0]]
      .transform("size")
)

suppressed_df = df.loc[row_group_sizes >= target_k].copy()

output_file = "suppressed_anonymized_data.csv"
suppressed_df.to_csv(output_file, index=False)

num_suppressed = num_rows - len(suppressed_df)

print("\nK=5 SUPPRESSION COMPLETE")
print(f"Rows removed:  {num_suppressed:,} ({num_suppressed / num_rows:.2%})")
print(f"Rows retained: {len(suppressed_df):,}")
print(f"Saved to:      {output_file}")

# Verify k-value
if not suppressed_df.empty:
    final_k = suppressed_df.groupby(data_cols, dropna=False).size().min()
    print(f"Verified minimum k: {final_k}")
else:
    print("No rows remain; k is undefined.")

# Display the smallest groups and their identifying combinations
# Used to determine k level when fewer than 20 unique groups exist.
# print("\nSMALLEST GROUPS")
# print(groups.sort_values("group_size").head(20).to_string(index=False))
#
# print("\nEND K-ANON REPORT")

