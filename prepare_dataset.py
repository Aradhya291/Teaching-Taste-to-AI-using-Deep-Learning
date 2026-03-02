import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# Path setup
INPUT_FILE = "data/ChemTastesDB_database.xlsx"
OUTPUT_FILE = "data/taste_data_5class.csv"
os.makedirs("data", exist_ok=True)

print("📖 Loading ChemTastesDB Excel file...")
df = pd.read_excel(INPUT_FILE)

# Normalize column names (strip spaces, lower-case)
df.columns = [c.strip().lower() for c in df.columns]

# Identify the correct columns
col_smiles = "canonical smiles" if "canonical smiles" in df.columns else "smiles"
col_taste = "taste" if "taste" in df.columns else "class taste"

print(f"✅ Using SMILES column: {col_smiles}")
print(f"✅ Using Taste column: {col_taste}")

# Keep only 5 major taste classes
target_classes = ["sweet", "bitter", "umami", "sour", "salty"]
df = df[df[col_taste].str.lower().isin(target_classes)].dropna(subset=[col_smiles])
df = df.drop_duplicates(subset=[col_smiles]).reset_index(drop=True)
print(f"✅ Found {len(df)} valid samples across 5 taste classes.")

# Generate RDKit descriptors
desc_funcs = [(d[0], d[1]) for d in Descriptors.descList]
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vals = []
    for _, func in desc_funcs:
        try:
            vals.append(func(mol))
        except Exception:
            vals.append(np.nan)
    return vals

print("⚙️ Generating molecular descriptors (this may take a few minutes)...")
rows, smiles_list, tastes = [], [], []
for _, row in df.iterrows():
    desc = featurize(row[col_smiles])
    if desc is not None and not np.isnan(desc).any():
        rows.append(desc)
        smiles_list.append(row[col_smiles])
        tastes.append(row[col_taste].lower())

desc_df = pd.DataFrame(rows, columns=[n for n, _ in desc_funcs])
final_df = pd.DataFrame({"smiles": smiles_list, "taste": tastes}).join(desc_df)
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved processed dataset to {OUTPUT_FILE}")
print(f"📊 Total samples: {len(final_df)}")
