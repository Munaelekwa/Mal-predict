import pandas as pd

# Load dataset
df = pd.read_csv("data/pubchem_data.csv")

# Step 1: Keep only relevant columns
relevant_columns = [
    'PUBCHEM_CID',
    'PUBCHEM_EXT_DATASOURCE_SMILES',
    'PUBCHEM_ACTIVITY_OUTCOME',
    'PUBCHEM_ACTIVITY_SCORE',
    'PubChem Standard Value',
    'Standard Type'
]
df = df[relevant_columns].copy()

# Convert standard_value to numeric, coercing errors
df['PubChem Standard Value'] = pd.to_numeric(df['PubChem Standard Value'], errors='coerce')

# Step 2: Drop rows with missing values in critical columns
df.dropna(subset=['PUBCHEM_EXT_DATASOURCE_SMILES', 'PubChem Standard Value', 'Standard Type'], inplace=True)

# Step 3: Filter standard types of interest
valid_types = ['IC50', 'EC50', 'ED50']
df = df[df['Standard Type'].isin(valid_types)]

# Step 4: Convert μM values to nM (assuming PubChem values are in μM)
df['standard_value_nM'] = df['PubChem Standard Value'] * 1000  # 1 μM = 1000 nM

# Step 5: Label as active/inactive
df['activity_class'] = df['standard_value_nM'].apply(lambda x: 1 if x < 1000 else 0)

# Optional: drop the old standard value column
df.drop(columns=['PubChem Standard Value'], inplace=True)

# Step 6: Rename SMILES column for clarity
df.rename(columns={'PUBCHEM_EXT_DATASOURCE_SMILES': 'canonical_smiles'}, inplace=True)

# Step 7: Final column selection and reordering
final_columns = ['PUBCHEM_CID', 'canonical_smiles', 'standard_value_nM', 'Standard Type', 'activity_class']
df = df[final_columns]

# Step 8: Print class distribution
print("Class Distribution:")
print(df['activity_class'].value_counts())

# Optional: Save cleaned data
df.to_csv("data/cleaned_pubchem_antimalarial.csv", index=False)
