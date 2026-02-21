# Aniti-Malarial-Activity-Prediction-Project
This Project aims to predict and classify the anti-malarial activity of compounds using machine learning techniques.

## Table of Contents  
- [Project Overview](#project-overview)  
- [Dataset Information](#dataset-information)  
- [Setup Instructions](#setup-instructions)  
  - [Prerequisites](#prerequisites)  
  - [Download and Installation](#download-and-installation)  
- [Featurisation](#featurisation)  
- [Model Building](#model-building)  
- [Model Evaluation](#model-evaluation)  
- [Results and Analysis](#results-and-analysis)
  - [Model Performance Summary](#model-performance-summary)  
  - [Areas of Improvement](#areas-of-improvement)
- [Live Project](#live-project) 
- [References](#references)

---

## Project Overview

Mal-predict is a machine learning–based project designed to support early-stage antimalarial drug discovery by predicting the activity of small molecules against Plasmodium falciparum. The model is trained on carefully curated bioactivity data from ChEMBL and PubChem and learns patterns from molecular structures associated with antimalarial activity. Using only SMILES representations as input, MalPredict classifies compounds as active or inactive and enables rapid virtual screening of large chemical libraries beyond traditional antimalarial datasets. Promising compounds identified by the model can then be prioritized for downstream structure-based analyses, including molecular docking and dynamics studies. 

## Dataset Information

This project uses curated bioactivity data for compounds tested against *Plasmodium falciparum*, obtained from the **ChEMBL** and **PubChem** databases.

Data from ChEMBL were retrieved by filtering for:
- Target organism: *Plasmodium falciparum*
- Bioactivity type: IC₅₀ values (in nM)

PubChem data were obtained from selected antimalarial-related assay IDs (AIDs) that reported quantitative activity measurements against *Plasmodium falciparum*. Only assays with clear activity readouts were included to maintain data quality.

After retrieval, datasets from both sources were merged, cleaned, and deduplicated to retain only unique compounds. Compounds were classified as **Active** or **Inactive** using an activity threshold of **1000 nM (1 µM)**. This threshold is widely used in antimalarial and early drug discovery research to distinguish compounds with meaningful biological activity from weak or inactive molecules.

### Dataset Summary

| Property | Value |
|--------|------|
| Total unique compounds | 33,208 |
| Activity threshold | IC₅₀ ≤ 1000 nM |
| Active compounds | 16,803 (50.6%) |
| Inactive compounds | 16,405 (49.4%) |
| Task type | Binary classification |

The resulting dataset is reasonably balanced, reducing the risk of class bias during model training and making it suitable for supervised machine learning.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python ≥ 3.9  
- RDKit  
- NumPy, Pandas, Scikit-learn  
- XGBoost  
- PyTorch & PyTorch Geometric (for FP-GNN)
- TensorFlow / Keras (for docking score regression)

### Download and Installation

```bash
git clone https://github.com/your-username/Aniti-Malarial-Activity-Prediction-Project.git
cd Aniti-Malarial-Activity-Prediction-Project
pip install -r requirements.txt
```

## Featurisation

Molecular structures were represented using **Morgan fingerprints** generated from canonical SMILES strings. Morgan fingerprints encode circular substructures around each atom into fixed-length binary vectors, allowing machine learning models to learn structure–activity relationships efficiently.

This representation was chosen because it:
- Captures local chemical environments relevant to bioactivity
- Is robust to small structural variations
- Produces fixed-length vectors suitable for classical machine learning models

For the deep learning experiments, molecules were additionally represented as **graph structures**, where atoms are treated as nodes and bonds as edges. This dual representation allowed the FP-GNN model to learn directly from molecular topology while also incorporating fingerprint-based features.

## Model Building

Multiple machine learning and deep learning models were explored to evaluate performance, robustness, and suitability for deployment:

- Random Forest (baseline)
- XGBoost (baseline)
- Optimized Random Forest
- Optimized XGBoost
- Stacking model (Random Forest + XGBoost)
- Fingerprint–Graph Neural Network (FP-GNN)
- Neural network for docking score regression

For the classical machine learning models, missing values were handled using mean imputation, and features were standardized prior to training. Stratified train–test splits were applied to preserve the class distribution of active and inactive compounds.

Hyperparameter optimization was performed using grid search (Random Forest) and randomized search (XGBoost) to improve model generalization.

## Model Evaluation

Classification models were evaluated using the following metrics:

- **Accuracy** – proportion of correctly classified compounds  
- **Precision** – proportion of predicted active compounds that are truly active  
- **Recall** – proportion of truly active compounds correctly identified  
- **F1-score** – harmonic mean of precision and recall  
- **AUROC** – ability of the model to distinguish active from inactive compounds across all thresholds  

AUROC was prioritized during model selection because it is threshold-independent and particularly suitable for virtual screening tasks, where ranking compounds by likelihood of activity is more important than a single classification cutoff.

The docking score regression model was evaluated using **mean squared error (MSE)** to measure how closely predicted docking scores matched observed values.


## Results and Analysis

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-score | AUROC |
|------|---------|-----------|--------|---------|-------|
| Random Forest (baseline) | 0.83 | 0.85 | 0.805 | 0.827 | **0.912** |
| XGBoost (baseline) | 0.796 | 0.815 | 0.773 | 0.793 | 0.877 |
| Random Forest (optimized) | 0.799 | 0.827 | 0.763 | 0.793 | 0.879 |
| XGBoost (optimized) | 0.771 | 0.788 | 0.751 | 0.769 | 0.847 |
| Stacking (RF + XGB) | 0.829 | 0.844 | 0.814 | 0.828 | **0.912** |

The Random Forest baseline and stacking models achieved the highest AUROC values, indicating strong discriminative performance. Considering model stability, interpretability, and ease of deployment, the **Random Forest baseline model** was selected for integration into the web application.

Although the FP-GNN model explored a more expressive deep learning architecture, it achieved lower performance (ROC AUC ≈ 0.65), suggesting that classical ensemble methods were better suited for this dataset and feature representation.

## Areas of Improvement

- Incorporation of experimental validation data for predicted hits  
- Improved interpretability of molecular features contributing to predictions  
- Expansion to multi-task learning across multiple malaria targets  
- Integration of ADMET-related endpoints for downstream compound prioritization  

## Live Project

The trained model is deployed as a web application (**MalPredict**), which allows users to:

- Input a single SMILES string and obtain an antimalarial activity prediction  
- Upload a CSV file containing multiple SMILES strings for batch screening  
- Download prediction results as a CSV file  
- Obtain predicted docking scores for four malaria-relevant protein targets  

**Live application:** *[mal-predict](https://mal-predict.streamlit.app/)*

## References

1. Lin M, Cai J, Wei Y, et al. MalariaFlow: A comprehensive deep learning platform for multistage phenotypic antimalarial drug discovery. *Eur J Med Chem*. 2024;277:116776. doi:10.1016/j.ejmech.2024.116776  
2. Gaulton A, Hersey A, Nowotka M, et al. The ChEMBL database in 2017. *Nucleic Acids Res*. 2017;45(D1):D945–D954.  
3. Kim S, Chen J, Cheng T, et al. PubChem in 2021: new data content and improved web interfaces. *Nucleic Acids Res*. 2021;49(D1):D1388–D1395.  



