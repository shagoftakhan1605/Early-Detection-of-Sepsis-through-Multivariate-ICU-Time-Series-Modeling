# Early Detection of Sepsis through Multivariate ICU Time-Series Modeling

This research presents a highly detailed machine learning and clinical analysis pipeline designed to predict sepsis using time-series physiological data. Through rigorous statistical modeling and biologically grounded interpretation, we aim to identify early indicators of sepsis onset and contribute to critical care decision-making systems.

## Objective
Sepsis is a dysregulated host response to infection, frequently progressing to life-threatening organ dysfunction. Early detection is essential. This project develops interpretable and high-performing models that learn from physiological biomarkers—mined from ICU time-series data—to predict the probability of sepsis onset, providing both predictive accuracy and clinical interpretability.

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

### 1. Data Preprocessing
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

### 2. Exploratory Data Analysis (EDA)

#### Correlation Matrix
- Insert: [correlation_heatmap_full.png]
- Key Observations:
  - **Creatinine ↔ BUN**: r ≈ 0.68 — indicative of acute kidney injury
  - **HR ↔ Lactate**: r ≈ 0.45 — signal for tissue hypoperfusion

#### Distribution Analysis
- Insert: [histogram_and_qqplot.png]
- Skewed distributions observed for Lactate, Bilirubin, WBC
- QQ-plots validated the requirement for non-parametric learning models

### 3. Feature Engineering
- Constructed Shock Index (HR/SBP) and rolling deltas in MAP to capture evolving cardiovascular collapse
- Time-windowed features derived to emulate clinician reasoning (e.g., MAP drop from t-2 to t)
- High-sparsity features (>25% nulls) filtered out
- Domain-informed selection retained:
  - **MAP** (hypotension)
  - **WBC** (immune activation)
  - **Platelets** (coagulopathy)
  - **Creatinine, BUN** (renal failure)

### 4. Model Architecture

#### Classifiers
- **Logistic Regression**: Interpretable linear baseline
- **Random Forest**: Optimal in terms of recall-precision tradeoff
- **XGBoost**: Best overall performance, especially for heterogeneous data
- **Naive Bayes** & **kNN**: Used for benchmarking; performed sub-optimally due to assumptions and scalability limits

#### Hyperparameter Optimization
- GridSearchCV tuned `n_estimators`, `max_depth`, `min_samples_leaf` for RF
- For XGBoost, explored:
  - `learning_rate`: [0.01, 0.05, 0.1, 0.3]
  - `scale_pos_weight`: [5, 10, 15] for imbalance calibration

### 5. Model Evaluation and Comparative Analysis
- Insert Confusion Matrices:
  - Logistic: [confusion_logistic.png]
  - Random Forest: [confusion_rf.png]
  - XGBoost: [confusion_xgb.png]

| Model          | Accuracy | Precision | Recall | F1 Score | AUC  |
|----------------|----------|-----------|--------|----------|------|
| Logistic Reg.  | 0.78     | 0.57      | 0.40   | 0.47     | 0.72 |
| Naive Bayes    | 0.73     | 0.46      | 0.31   | 0.37     | 0.68 |
| k-Nearest Neigh| 0.76     | 0.51      | 0.36   | 0.42     | 0.70 |
| Random Forest  | **0.95** | **0.91**  | **0.94**| **0.933**| **0.95** |
| **XGBoost**    | 0.88     | 0.73      | 0.69   | 0.71     | 0.84 |

### External Cohort Validation (Domain Shift)
- When tested on unseen hospital system (e.g., System C), Random Forest F1-score dropped to **0.14**
- Indicates significant domain shift due to lab ordering patterns, patient demographics, and instrumentation
- Reinforces need for **domain adaptation, federated learning, and model calibration** in real-world deployment

## Biological Interpretation of Findings

[Section unchanged — continues as is with plot references and physiological insight.]

## Conclusion
This study successfully built and evaluated a clinically grounded, biologically interpretable sepsis prediction model with strong internal performance and generalization challenges on cross-institutional data. Random Forest emerged as the best classifier, learning discriminative patterns in early physiological decompensation, with results validating known sepsis pathophysiology.

## Future Directions
- Temporal deep learning (LSTM, Transformer) for dynamic prediction
- Domain adaptation strategies (CORAL, TTA, FedAvg)
- Structured alert protocol evaluation with ICU clinicians
- External deployment with real-time hospital data streams

## [Other sections remain unchanged — Setup, File Structure, Author, etc.]


