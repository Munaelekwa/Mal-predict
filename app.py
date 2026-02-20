import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
import joblib
import tensorflow as tf

# Load models
@st.cache_resource
def load_activity_model():
    return joblib.load("models/random_forest_model.pkl")  # RF trained on 2048 FP + 7 descriptors

@st.cache_resource
def load_affinity_model():
    return tf.keras.models.load_model(
        "models/docking_score_prediction_model.h5",
        custom_objects={'mse': tf.keras.metrics.MeanSquaredError()}
    )

activity_model = load_activity_model()
affinity_model = load_affinity_model()

# Parameters for featurization
fingerprint_size = 2048
descriptor_names = [
    'MolWt', 'MolLogP', 'NumRotatableBonds',
    'NumHAcceptors', 'NumHDonors', 'TPSA', 'RingCount'
]

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fingerprint_size)

def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.array([np.nan] * len(descriptor_names))
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.RingCount(mol)
    ])

def featurize_activity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_arr = np.zeros((fingerprint_size,), dtype=int)
    fp = morgan_gen.GetFingerprint(mol)
    ConvertToNumpyArray(fp, fp_arr)
    desc_arr = calc_descriptors(smiles)
    if np.any(np.isnan(desc_arr)):
        return None
    return np.concatenate([fp_arr, desc_arr])

def featurize_affinity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_arr = np.zeros((fingerprint_size,), dtype=int)
    fp = morgan_gen.GetFingerprint(mol)
    ConvertToNumpyArray(fp, fp_arr)
    return fp_arr

protein_order = ["9NSR", "3EBH", "8EM8", "3I65"]
protein_descriptions = {
    "9NSR": "Bifunctional farnesyl/geranylgeranyl pyrophosphate synthase (FPPS/GGPPS) from Plasmodium falciparum, a key enzyme in isoprenoid biosynthesis and validated antimalarial drug target.",
    "3EBH": "M1 alanylaminopeptidase from Plasmodium falciparum, involved in hemoglobin degradation and parasite survival inside red blood cells.",
    "8EM8": "cGMP-dependent protein kinase (PKG) from Plasmodium falciparum, regulating signaling pathways essential for parasite egress and transmission.",
    "3I65": "Dihydroorotate dehydrogenase (DHODH) from Plasmodium falciparum, a central enzyme in pyrimidine biosynthesis and a well-known antimalarial target."
}

def predict_activity(smiles):
    features = featurize_activity(smiles)
    if features is None:
        st.error("Invalid SMILES or descriptor calculation failed for activity prediction.")
        return None
    X = features.reshape(1, -1)
    pred_label = activity_model.predict(X)[0]
    pred_proba = activity_model.predict_proba(X)[0][1]
    label = "Active" if pred_label == 1 else "Inactive"
    return label, pred_proba

def predict_affinity(smiles):
    features = featurize_affinity(smiles)
    if features is None:
        st.error("Invalid SMILES or fingerprint calculation failed for affinity prediction.")
        return None
    X = features.reshape(1, -1).astype(np.float32)
    preds = affinity_model.predict(X)[0]
    return dict(zip(protein_order, preds))

# --- Streamlit UI ---
st.set_page_config(page_title="Mal-predict", page_icon="🧪", layout="wide")

st.markdown("""
<style>
h1 {
    color: #2e7d32;
    font-weight: 700;
    font-size: 36px;
}
h2 {
    color: #388e3c;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Antimalarial Activity & Binding Affinity Predictor")
st.markdown("""
This tool predicts the antimalarial **activity** of small molecules and estimates their **binding affinities** against four validated *Plasmodium falciparum* drug targets, based on SMILES strings.
""")

st.markdown("### Protein targets used in docking")
st.markdown("""
**9NSR – FPPS/GGPPS (Isoprenoid biosynthesis)**
Crystal structure of bifunctional farnesyl/geranylgeranyl pyrophosphate synthase (FPPS/GGPPS) from *Plasmodium falciparum* in complex with MMV019313.

**3EBH – M1 alanylaminopeptidase (Hemoglobin degradation)**
Structure of the M1 alanylaminopeptidase from malaria complexed with bestatin, involved in hemoglobin degradation and parasite survival.

**8EM8 – PKG (cGMP-dependent protein kinase)**
Co-crystal structure of the cGMP-dependent protein kinase PKG from *Plasmodium falciparum* in complex with inhibitor RY-1-165, a regulator of critical signaling events.

**3I65 – DHODH (Pyrimidine biosynthesis)**
*Plasmodium falciparum* dihydroorotate dehydrogenase bound with triazolopyrimidine-based inhibitor DSM1, a key enzyme in the pyrimidine biosynthesis pathway.
""")

st.markdown("---")

tab_single, tab_batch = st.tabs(["Single compound prediction", "Batch prediction (CSV)"])

# ------------------------ SINGLE SMILES TAB ------------------------ #

with tab_single:
    st.subheader("Single compound prediction")

    smiles_input = st.text_input(
        "Enter SMILES string",
        placeholder="e.g. CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    )

    if st.button("Predict for this compound"):
        if not smiles_input.strip():
            st.warning("Please enter a valid SMILES string.")
        else:
            with st.spinner("Running predictions..."):
                label, prob = predict_activity(smiles_input)
                affinities = predict_affinity(smiles_input)

            if label is None:
                st.error("Could not featurize SMILES for activity prediction.")
            else:
                st.markdown("#### Antimalarial activity")
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("Predicted class", label)
                with c2:
                    st.progress(prob if label == "Active" else 1 - prob)
                st.write(f"Estimated probability of being **Active**: `{prob:.4f}`")

            if affinities is None:
                st.error("Could not featurize SMILES for binding affinity prediction.")
            else:
                st.markdown("#### Binding affinity predictions (docking scores)")
                df_aff = pd.DataFrame([
                    {
                        "Protein code": code,
                        "Target": protein_descriptions[code].split(" – ")[0] if "–" in protein_descriptions[code] else "",
                        "Predicted binding affinity": f"{affinities[code]:.3f}"
                    }
                    for code in protein_order
                ])
                st.table(df_aff)

# ------------------------ BATCH CSV TAB ------------------------ #

with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.markdown("""
Upload a CSV file containing a column named `smiles`.
For each compound, the app will predict:
- Antimalarial activity (Active/Inactive + probability), and
- Binding affinities to the four protein targets.
""")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_in = None

        if df_in is not None:
            if "smiles" not in df_in.columns:
                st.error("CSV must contain a column named 'smiles'.")
            else:
                if st.button("Run batch prediction"):
                    results = []
                    with st.spinner("Running batch predictions..."):
                        for idx, row in df_in.iterrows():
                            smi = row["smiles"]
                            label, prob = predict_activity(smi)
                            affinities = predict_affinity(smi)

                            if (label is None) or (affinities is None):
                                results.append({
                                    "smiles": smi,
                                    "activity_label": None,
                                    "activity_prob_active": None,
                                    "affinity_9NSR": None,
                                    "affinity_3EBH": None,
                                    "affinity_8EM8": None,
                                    "affinity_3I65": None,
                                })
                            else:
                                results.append({
                                    "smiles": smi,
                                    "activity_label": label,
                                    "activity_prob_active": prob,
                                    "affinity_9NSR": affinities["9NSR"],
                                    "affinity_3EBH": affinities["3EBH"],
                                    "affinity_8EM8": affinities["8EM8"],
                                    "affinity_3I65": affinities["3I65"],
                                })

                    df_res = pd.DataFrame(results)
                    st.markdown("#### Batch prediction results")
                    st.dataframe(df_res)

                    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download results as CSV",
                        data=csv_bytes,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )

st.markdown("---")
st.markdown("""
**Interpretation note:**
Docking scores are typically negative (strong binding) to positive (weak/no binding).
More negative values suggest stronger predicted binding to the target.

This tool is for research and screening support, not a substitute for experimental validation.
By Munachiso Elekwa (2026)
""")
