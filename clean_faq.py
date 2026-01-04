import pandas as pd

# Load your file
df = pd.read_csv("Library FAQ.csv")

# Keep only relevant columns
df = df[["id", "question", "answer"]]

# Save cleaned version
df.to_csv("library_faq_clean.csv", index=False)

print("Saved library_faq_clean.csv with", len(df), "rows")
