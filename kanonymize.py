import pandas as pd

df = pd.read_csv("raw_data.csv", dtype=str, keep_default_na=False)

# Remove postal code and city columns.
df = df.drop(columns=["postalCode", "city"], errors="ignore")

# Generalize the last digit of birth years: 1991 -> 199*
if "YoB" in df.columns:
    df["YoB"] = df["YoB"].str.replace(
        r"^(\d{3})\d$", r"\1*", regex=True
    )

# Generalize college education 
if "LoE" in df.columns:
    df["LoE"] = df["LoE"].replace({
        "a": "college or above",
        "b": "college or above",
        "m": "college or above",
        "p": "college or above",
        "p_se": "college or above",
        "p_oth": "college or above",

        "hs": "no college education",
        "jhs": "no college education",
        "el": "no college education",
        "none": "no college education",

        "other": "Unknown",
        "": "Unknown",
        "null": "Unknown",
    }).fillna("Unknown")

# Save the modified data to a separate file.
df.to_csv("anonymized_data.csv", index=False)

print(df)
print(f"Saved {len(df):,} rows to anonymized_data.csv")