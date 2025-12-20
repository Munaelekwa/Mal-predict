import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
import joblib
import tensorflow as tf

# ------------------------ MODELS ------------------------ #

@st.cache_resource
def load_activity_model():
    return joblib.load("models/random_forest_model.pkl")

@st.cache_resource
def load_affinity_model():
    return tf.keras.models.load_model(
        "models/docking_score_prediction_model.h5",
        custom_objects={'mse': tf.keras.metrics.MeanSquaredError()}
    )

activity_model = load_activity_model()
affinity_model = load_affinity_model()

# ------------------------ FEATURIZATION ------------------------ #

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

# ------------------------ PREDICTION HELPERS ------------------------ #

protein_order = ["9NSR", "3EBH", "8EM8", "3I65"]
protein_descriptions = {
    "9NSR": "FPPS/GGPPS – isoprenoid biosynthesis enzyme from Plasmodium falciparum.",
    "3EBH": "M1 alanylaminopeptidase – involved in hemoglobin degradation.",
    "8EM8": "PKG – cGMP‑dependent protein kinase controlling key signaling events.",
    "3I65": "DHODH – central enzyme in pyrimidine biosynthesis.",
}

def predict_activity_single(smiles):
    features = featurize_activity(smiles)
    if features is None:
        return None, None
    X = features.reshape(1, -1)
    pred_label = activity_model.predict(X)[0]
    proba = activity_model.predict_proba(X)[0][1]
    label = "Active" if pred_label == 1 else "Inactive"
    return label, proba

def predict_affinity_single(smiles):
    features = featurize_affinity(smiles)
    if features is None:
        return None
    X = features.reshape(1, -1).astype(np.float32)
    preds = affinity_model.predict(X)[0]
    return dict(zip(protein_order, preds))

# ------------------------ STREAMLIT UI ------------------------ #

st.set_page_config(
    page_title="Antimalarial Activity & Binding Affinity Predictor",
    page_icon="🧪",
    layout="wide"
)

# Global styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f3f7ff 0%, #f9fffd 100%);
    }
    .main {
        background-color: transparent;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
        border: 1px solid #edf2f7;
    }
    h1 {
        color: #1b5e20;
        font-weight: 800;
    }
    h2 {
        color: #2e7d32;
        font-weight: 700;
    }
    h4 {
        color: #1b5e20;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #2e7d32;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .protein-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }
    .p-9nsr { background-color: #1e88e5; }
    .p-3ebh { background-color: #8e24aa; }
    .p-8em8 { background-color: #fb8c00; }
    .p-3i65 { background-color: #43a047; }
    .credits {
        font-size: 0.9rem;
        color: #555;
        text-align: center;
        margin-top: 1.5rem;
    }
    .divider-soft {
        border-top: 1px solid #e0e0e0;
        margin: 1.2rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 Antimalarial Activity & Binding Affinity Predictor")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
        This tool predicts the antimalarial **activity** of small molecules and estimates their **binding affinities**
        against four validated *Plasmodium falciparum* drug targets using SMILES input.
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Protein targets used in docking</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <span class="protein-badge p-9nsr">9NSR · FPPS/GGPPS</span>
            Bifunctional farnesyl/geranylgeranyl pyrophosphate synthase, a key enzyme in **isoprenoid biosynthesis** and a validated antimalarial target.
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <span class="protein-badge p-3ebh">3EBH · M1 aminopeptidase</span>
            M1 alanylaminopeptidase involved in **hemoglobin degradation** and parasite survival inside red blood cells.
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <span class="protein-badge p-8em8">8EM8 · PKG</span>
            cGMP‑dependent protein kinase **PKG**, regulating signaling pathways essential for parasite egress and transmission.
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <span class="protein-badge p-3i65">3I65 · DHODH</span>
            Dihydroorotate dehydrogenase, a central enzyme in **pyrimidine biosynthesis** and a well‑known antimalarial target.
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)



tab_single, tab_batch = st.tabs(["🔹 Single compound", "📁 Batch prediction (CSV)"])

# ------------------------ SINGLE SMILES TAB ------------------------ #

with tab_single:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Single compound prediction")

    smiles_input = st.text_input(
        "Enter SMILES string",
        placeholder="e.g. CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    )

    if st.button("Run prediction for this compound"):
        if not smiles_input.strip():
            st.warning("Please enter a valid SMILES string.")
        else:
            with st.spinner("Running activity and binding affinity predictions..."):
                label, prob = predict_activity_single(smiles_input)
                affinities = predict_affinity_single(smiles_input)

            if label is None:
                st.error("Could not featurize SMILES for activity prediction.")
            else:
                st.markdown("#### Antimalarial activity")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Predicted class", label)
                with col2:
                    st.progress(prob if label == "Active" else 1 - prob)
                st.write(f"Estimated probability of being **Active**: `{prob:.4f}`")

            if affinities is None:
                st.error("Could not featurize SMILES for binding affinity prediction.")
            else:
                st.markdown("#### Binding affinity predictions (docking scores)")
                df_aff = pd.DataFrame([
                    {
                        "Protein code": code,
                        "Target (short description)": protein_descriptions[code],
                        "Predicted binding affinity": f"{affinities[code]:.3f}",
                    }
                    for code in protein_order
                ])
                st.dataframe(df_aff, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ BATCH CSV TAB ------------------------ #

with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Batch prediction from CSV")

    st.markdown(
        """
        Upload a CSV file containing a column named `smiles`.
        For each compound, the app will predict:
        - Antimalarial activity (Active/Inactive + probability), and
        - Binding affinities to the four protein targets.
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_in = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_in = None

        if df_in is not None:
            if "smiles" not in df_in.columns:
                st.error("CSV must contain a column named `smiles`.")
            else:
                if st.button("Run batch prediction"):
                    results = []
                    with st.spinner("Running batch predictions..."):
                        for _, row in df_in.iterrows():
                            smi = row["smiles"]
                            label, prob = predict_activity_single(smi)
                            affinities = predict_affinity_single(smi)

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
                    st.dataframe(df_res, width="stretch")

                    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download results as CSV",
                        data=csv_bytes,
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """
**Interpretation note**
Docking scores are typically negative (strong binding) to positive (weak/no binding).
More negative values suggest stronger predicted binding to the target.

<div class="credits">
Built by <strong>Munachi Elekwa (2025)</strong> · Antimalarial ML Screening Tool
</div>
""",
    unsafe_allow_html=True,
)
