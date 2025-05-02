# 🧬 Early Detection of Sepsis through Multivariate ICU Time-Series Modeling

This research presents a highly detailed machine learning and clinical analysis pipeline designed to predict sepsis using time-series physiological data. Through rigorous statistical modeling and biologically grounded interpretation, we aim to identify early indicators of sepsis onset and contribute to critical care decision-making systems.

---
## Research Problem and Motivation

Sepsis, a life-threatening organ dysfunction due to a dysregulated host response to infection, contributes to over 11 million global deaths annually. The urgent challenge lies in early detection — current tools often diagnose sepsis only after organ dysfunction has set in. Sepsis is a dysregulated host response to infection, frequently progressing to life-threatening organ dysfunction. Early detection is essential. This project develops interpretable and high-performing models that learn from physiological biomarkers—mined from ICU time-series data—to predict the probability of sepsis onset, providing both predictive accuracy and clinical interpretability.

This project aims to close that gap by asking:

1. **Can we predict sepsis onset ≥6 hours before clinical diagnosis using ICU time-series data?**  
2. **What clinical biomarkers consistently precede sepsis across diverse hospital systems?**  
3. **How do predictive models behave across domains with differing data sparsity and recording frequency?**  
4. **Can missingness patterns (entry density) reveal systemic bias or hidden clinical practices that affect model generalizability?**

---

## Biological and Clinical Rationale

Sepsis pathogenesis involves a cascade of rapidly evolving physiological breakdowns. Our focus lies in **modeling host responses** through vital signs and biomarkers:

- **Hemodynamic collapse:** captured via MAP and DBP  
- **Renal failure:** tracked by Creatinine and BUN  
- **Immune dysregulation:** modeled through WBC variability  
- **Coagulopathy:** inferred from declining Platelets

These features are not just statistically significant but biologically interpretable, linking directly to the Sepsis-3 criteria and SOFA score dimensions.

---

## Prediction Horizon and Clinical Timeline
The model was explicitly designed to predict sepsis onset **at least 6 hours prior to clinical diagnosis**. This predictive window aligns with real-world triage and early-warning requirements in intensive care units, enabling physicians to initiate antibiotics and supportive therapy during the reversible phase of sepsis pathophysiology.

## Dataset
- **Source**: PhysioNet 2019 Challenge Dataset
- **Records**: ICU patient records in hourly resolution from >20,000 patients
- **Clinical Features**:
  - **Vital signs**: HR, O2Sat, Temp, SBP, DBP, MAP, Resp
  - **Lab values**: Creatinine, Lactate, Glucose, WBC, Platelets, Bilirubin, Troponin
  - **Metadata**: Age, Gender, ICU stay metrics, Sepsis label (binary)
- **Hospital Systems**: A, B, and C — providing structurally distinct data for generalization testing
- **Missingness**: Lab data sparsity addressed via domain-specific imputation

## Methodology

### Data Preprocessing
- Combined and restructured patient-wise time-series files across hospitals.
- **Imputation Strategy**:
  - *Bidirectional Imputation (bfill/ffill)* for vital signs and frequently sampled labs
  - *KNN Imputation* for sparse labs using Euclidean distance between similar ICU profiles
- **Normalization**:
  - *Logarithmic transformation* for biomarkers like Lactate and Creatinine
  - *Yeo-Johnson transformation* for variables with zero or negative values
  - Standardized all continuous features using z-score
- **Categorical Processing**:
  - One-hot encoding of Gender, ICU Type, and hospital ID
- **Class Balancing**:
  - Applied undersampling of non-septic records to maintain temporal context while addressing class imbalance
- **Feature Pruning:** Removed:
  - Variables with >25% nulls
  - Biologically redundant or implausible variables
 
---
### Biologically Informed Feature Selection

Variables selected based on:
- Pathophysiological importance
- Availability across institutions

| Category         | Key Features                        |
|------------------|--------------------------------------|
| Hemodynamic      | MAP, SBP, DBP                        |
| Renal Function   | Creatinine, BUN                      |
| Coagulation      | Platelets                            |
| Immune Response  | WBC                                  |
| Oxygenation      | SpO₂, FiO₂                           |

---
### Exploratory Data Analysis (EDA)

#### Correlation Matrix
![5](https://github.com/user-attachments/assets/8aa868d4-b37f-4890-bd69-291118866c46)

- Key Observations:
  - **Creatinine ↔ BUN**: r ≈ 0.68 — indicative of acute kidney injury
  - **HR ↔ Lactate**: r ≈ 0.45 — signal for tissue hypoperfusion

#### Distribution Analysis
![3](https://github.com/user-attachments/assets/c305bcf5-6f97-4a99-b4db-8736ca9a97bd)

- Skewed distributions observed for Lactate, Bilirubin, WBC
- QQ-plots validated the requirement for non-parametric learning models

### Feature Engineering
- Constructed Shock Index (HR/SBP) and rolling deltas in MAP to capture evolving cardiovascular collapse
- Time-windowed features derived to emulate clinician reasoning (e.g., MAP drop from t-2 to t)
- High-sparsity features (>25% nulls) filtered out
- Domain-informed selection retained:
  - **MAP** (hypotension)
  - **WBC** (immune activation)
  - **Platelets** (coagulopathy)
  - **Creatinine, BUN** (renal failure)
---

## Modeling Framework

- **Algorithms:**  
  - Random Forest (tuned to n=300)  
  - XGBoost  
  - Logistic Regression  
  - Naive Bayes  
  - k-Nearest Neighbors  

- **Evaluation Metrics:**  
  - ROC-AUC, F1-score, Precision, Recall  
  - Confusion Matrices for sensitivity-specificity analysis  

- **Training Strategy:**  
  - Stratified 80/20 train-test split  
  - Cross-validation with attention to class imbalance (undersampling majority class)
---
## Hyperparameter Optimization
- GridSearchCV tuned `n_estimators`, `max_depth`, `min_samples_leaf` for RF
- For XGBoost, explored:
  - `learning_rate`: [0.01, 0.05, 0.1, 0.3]
  - `scale_pos_weight`: [5, 10, 15] for imbalance calibration
---
## Model Evaluation and Comparative Analysis
  - Logistic: ![download](https://github.com/user-attachments/assets/a5aa4a90-bd8e-4ccb-84c6-cbcfeff053c4)
  - Random Forest: ![download](https://github.com/user-attachments/assets/bfc1b157-6c60-45ca-b47f-d5e9e9be1e31)
  - Naive Bayes Classifier: ![download](https://github.com/user-attachments/assets/f549bd6e-61eb-4795-842e-90575e56eaef)
  - KNN Classifier: ![download](https://github.com/user-attachments/assets/e4e4e701-27e6-46af-b411-e6fdee7fefe4)
  - XGBoost: ![download](https://github.com/user-attachments/assets/1de24c8d-885e-42b1-963c-4ba2bc9e7558)


| Model          | Accuracy | Precision | Recall | F1 Score | AUC  |
|----------------|----------|-----------|--------|----------|------|
| Logistic Reg.  | 0.78     | 0.57      | 0.40   | 0.47     | 0.72 |
| Naive Bayes    | 0.73     | 0.46      | 0.31   | 0.37     | 0.68 |
| k-Nearest Neigh| 0.76     | 0.51      | 0.36   | 0.42     | 0.70 |
| Random Forest  | **0.95** | **0.91**  | **0.94**| **0.933**| **0.95** |
| **XGBoost**    | 0.88     | 0.73      | 0.69   | 0.71     | 0.84 |

---

### Performance on Hospital B (Domain Shift)

| Metric       | Value         |
|--------------|---------------|
| ROC-AUC      | 0.58          |
| F1 Score     | 14.1%         |

> ⚠️ Dramatic drop in generalizability emphasizes the need for domain adaptation techniques.

### External Cohort Validation (Domain Shift)
- When tested on unseen hospital system (e.g., System C), Random Forest F1-score dropped to **0.14**
- Indicates significant domain shift due to lab ordering patterns, patient demographics, and instrumentation
- Reinforces need for **domain adaptation, federated learning, and model calibration** in real-world deployment

---

## Interpretability and Feature Insights

- **MAP** (↓): predictor of circulatory collapse  
- **Creatinine/BUN** (↑): markers of renal stress preceding systemic infection  
- **Platelets** (↓): reflect coagulation cascade breakdown  
- **WBC** (biphasic): hyper- and hypo-responsiveness seen in septic phases  

These insights align with known clinical progression of early sepsis, reinforcing the pipeline’s interpretability and biological validity.

---

## Visual Summaries

- **Entry Density Maps**: Show data availability variation across institutions  
- **QQ & Histogram Plots**: Confirm non-Gaussian distributions, justifying transformations  
- **Confusion Matrices**: Validate sensitivity-prioritized classification  
- **Correlation Heatmaps**: Reveal variable co-dependencies and latent redundancies


---

## Contributions

- ✅ Built a biologically consistent, interpretable pipeline for early sepsis detection  
- ✅ Identified robust predictors across vital categories  
- ✅ Demonstrated severe cross-hospital model decay, motivating domain-robust strategies  
- ✅ Produced a full evaluation suite including density diagnostics and label-stratified confusion plots
  

## Future Directions
- Temporal deep learning (LSTM, Transformer) for dynamic prediction
- Domain adaptation strategies (CORAL, TTA, FedAvg)
- Structured alert protocol evaluation with ICU clinicians
- External deployment with real-time hospital data streams
