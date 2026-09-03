import pandas as pd

INPUT_PATH = "data/SMSSpamCollection"
OUTPUT_PATH = "data/spam.csv"

# Read the original UCI dataset
df = pd.read_csv(
    INPUT_PATH,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Save as CSV
df.to_csv(OUTPUT_PATH, index=False)

print("Dataset converted successfully!")
print("Total messages:", len(df))
print()
print(df.head())
print()
print("Label distribution:")
print(df["label"].value_counts())