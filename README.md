# Early Detection of Sepsis through Multivariate ICU Time-Series Modeling

This project builds and compares machine learning classifiers that predict sepsis from hourly ICU data, and then tests how well a model trained on one hospital transfers to another. All code is in [`sepsis_prediction.ipynb`](sepsis_prediction.ipynb) (Google Colab).

---

## Research Questions

1. Can routinely recorded vital signs and lab values predict the sepsis label in hourly ICU records?
2. Which classical models perform best on this task?
3. How much does performance drop when a model is applied to a different hospital?

---

## Dataset

* **Source:** PhysioNet/Computing in Cardiology Challenge 2019 (Reyna et al., 2020), loaded as a combined CSV from Kaggle. A copy of the archive used is linked in [`Dataset Zip File`](Dataset%20Zip%20File).
* **Structure:** hourly records per patient from **two hospital systems**. In the notebook they are separated by `Patient_ID` format: one hospital is used for training and testing, the other (20,000 patients) for cross-hospital testing.
* **Label:** `SepsisLabel`. In the PhysioNet 2019 data this label is already shifted to become positive **6 hours before** clinical sepsis onset, so the task is early prediction by design of the labels.

Reference: Reyna, M. A., et al. (2020). Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210-217. https://doi.org/10.1097/CCM.0000000000004145

---

## Method (as implemented in the notebook)

### Feature selection
* Columns were dropped for redundancy or limited availability (for example SBP and DBP, since MAP is kept; BaseExcess, HCO3, pH, PaCO2, EtCO2).
* After imputation, columns with more than 25% missing values were dropped (TroponinI, Bilirubin, AST, Lactate, SaO2, FiO2, Unit).
* **Final features:** Hour, HR, O2Sat, Temp, MAP, Resp, BUN, Chloride, Creatinine, Glucose, Hct, Hgb, WBC, Platelets, Age, Gender (one-hot), HospAdmTime, ICULOS.

### Preprocessing
* **Imputation:** backward fill then forward fill within each patient
* **Transformation:** several transforms were compared with diagnostic and Q-Q plots (Yeo-Johnson, power, reciprocal, log). Log transform was kept for MAP, BUN, Creatinine, Glucose, WBC and Platelets.
* **Scaling:** z-score standardisation of continuous features
* Remaining rows with missing values were removed

### Class imbalance
* Training hospital: 15,694 sepsis-positive rows and 760,125 negative rows
* Negative rows were randomly undersampled to a 2:1 ratio

### Models
Random split of rows, 80% train and 20% test:
* Random Forest (n_estimators 100, 200 and 300 tried; 300 kept)
* Gaussian Naive Bayes
* k-Nearest Neighbours (k = 5, 8 and 10 tried; 10 kept)
* Logistic Regression
* XGBoost (max depth 5, eta 0.3, 100 rounds)

---

## Results

### Same hospital (20% held-out rows)

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| **Random Forest** | **0.95** | **0.92** | **0.95** | **0.93** | **0.95** |
| XGBoost | 0.86 | 0.83 | 0.73 | 0.77 | 0.83 |
| k-Nearest Neighbours | 0.82 | 0.77 | 0.64 | 0.70 | 0.77 |
| Naive Bayes | 0.74 | 0.69 | 0.40 | 0.51 | 0.66 |
| Logistic Regression | 0.74 | 0.71 | 0.37 | 0.48 | 0.65 |

AUC here is computed from predicted class labels, not probabilities.

Confusion matrices:
  - Logistic: ![download](https://github.com/user-attachments/assets/a5aa4a90-bd8e-4ccb-84c6-cbcfeff053c4)
  - Random Forest: ![download](https://github.com/user-attachments/assets/bfc1b157-6c60-45ca-b47f-d5e9e9be1e31)
  - Naive Bayes: ![download](https://github.com/user-attachments/assets/f549bd6e-61eb-4795-842e-90575e56eaef)
  - KNN: ![download](https://github.com/user-attachments/assets/e4e4e701-27e6-46af-b411-e6fdee7fefe4)
  - XGBoost: ![download](https://github.com/user-attachments/assets/1de24c8d-885e-42b1-963c-4ba2bc9e7558)

### Second hospital (cross-hospital test)

| Metric | Value |
| --- | --- |
| Accuracy | 0.83 |
| Precision | 0.08 |
| Recall | 0.54 |
| F1 | 0.14 |
| AUC | 0.69 |

Performance drops sharply on the second hospital, mainly through very low precision.

**Note:** in the notebook, this test uses the variable `model`, which at that point holds the **Logistic Regression** model (it was reassigned after Random Forest). So this result reflects Logistic Regression, not Random Forest. See limitations.

---

## Exploratory Analysis

Correlation matrix:
![5](https://github.com/user-attachments/assets/8aa868d4-b37f-4890-bd69-291118866c46)

Distribution analysis:
![3](https://github.com/user-attachments/assets/c305bcf5-6f97-4a99-b4db-8736ca9a97bd)

Several lab variables are strongly skewed, which motivated the log transform.

---

## Limitations

* **Row-level split.** Train and test sets were split by row, not by patient, so hours from the same patient can appear in both. This likely inflates same-hospital scores, especially for Random Forest.
* **Cross-hospital test model.** The second-hospital result came from Logistic Regression, not the best model.
* **Separate scaling on the test hospital.** The second hospital was standardised with its own scaler instead of the training scaler.
* **Hourly rows treated independently.** No temporal features (trends, deltas) were used, so the models do not use the time-series structure.
* **Threshold-based AUC.** AUC was computed from hard predictions, which understates ranking ability.

---

## Next Steps

* Split by patient (for example `GroupShuffleSplit` on `Patient_ID`) and rerun all models
* Re-test the second hospital with the trained Random Forest, using the training scaler
* Add temporal features such as rolling means and changes in MAP and HR
* Report AUC from predicted probabilities and add the PhysioNet utility score
* Try sequence models (LSTM, Transformer) and domain adaptation for the cross-hospital gap

---

## Tools

Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, Matplotlib, Seaborn
