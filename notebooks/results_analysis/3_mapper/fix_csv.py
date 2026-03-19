import csv, shutil, os

src = "results_summary_all.csv"

# Backup original
if not os.path.exists(src + ".bak"):
    shutil.copy(src, src + ".bak")

with open(src, "rb") as f:
    raw = f.read()

# Decode
text = raw.decode("utf-8")

# --- Fix 1: missing newline between restaurant/mistral-large and sisu rows ---
# The last field of restaurant/mistral-large ends with "0.5000" and then "sisu," starts
# without a newline. Find all occurrences and insert newline.
import re
# Pattern: a decimal value immediately followed by "sisu," without a newline
text = re.sub(r"([\d\.]+)(sisu,)", r"\1\n\2", text)

# Also fix similar joins with other dataset names just in case
for ds in ["tax,", "hospital,", "restaurant,"]:
    text = re.sub(r"([\d\.]+)(" + ds + ")", r"\1\n\2", text)

# Write fixed text back
with open(src, "w", newline="", encoding="utf-8") as f:
    f.write(text)

# --- Fix 2: read all rows and strip trailing empty columns, standardize to 42 cols ---
with open(src, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    ncols = len(header)
    print(f"Header has {ncols} columns")
    rows = []
    for row in reader:
        # Strip trailing empty fields
        while len(row) > ncols and row[-1] == "":
            row = row[:-1]
        rows.append(row)

bad = [(i+2, len(r)) for i, r in enumerate(rows) if len(r) != ncols]
print(f"Bad rows: {bad}")

# --- Fix 3: add missing sisu/llm/mistral-large-2512 failed row if absent ---
has_sisu_llm_mistral_large = any(
    r[0] == "sisu" and r[2] == "llm" and "mistral-large" in r[3]
    for r in rows if len(r) > 3
)
print(f"sisu/llm/mistral-large present: {has_sisu_llm_mistral_large}")

if not has_sisu_llm_mistral_large:
    failed_row = (
        ["sisu", "LLM | mistral-large | with_examples_and_table", "llm",
         "mistral-large-2512", "with_examples_and_table", "N/A", "True",
         "", "42", "17", "N/A", "failed"]
        + [""] * (ncols - 12)
    )
    # Insert after last sisu/llm row
    insert_at = None
    for i, r in enumerate(rows):
        if len(r) > 2 and r[0] == "sisu" and r[2] == "llm":
            insert_at = i + 1
    if insert_at is not None:
        rows.insert(insert_at, failed_row)
    else:
        rows.append(failed_row)
    print("Inserted missing sisu/llm/mistral-large row")

bad_after = [(i+2, len(r), r[0] if r else "") for i, r in enumerate(rows) if len(r) != ncols]
print(f"Bad rows after fix: {bad_after}")

# --- Write back ---
with open(src, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Done. Total rows written: {len(rows)}")

# --- Verify with pandas ---
import pandas as pd
df = pd.read_csv(src)
print(f"Pandas read OK: {len(df)} rows, {len(df.columns)} cols")
sisu_rows = df[df["dataset"] == "sisu"][["implementation", "model", "status"]]
print("SISU rows:")
print(sisu_rows.to_string())
