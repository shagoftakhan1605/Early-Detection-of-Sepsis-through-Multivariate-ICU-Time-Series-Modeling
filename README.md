# Early Detection of Sepsis through Multivariate ICU Time-Series Modeling

This project presents a detailed machine learning pipeline to predict sepsis using physiological data from the PhysioNet database. It includes data preprocessing, imputation, model training, evaluation, and visualization, ultimately aiming to build an effective early warning system for sepsis.

## Objective
Sepsis is a critical condition that arises from the body's extreme response to infection and can lead to tissue damage, organ failure, and death. Early detection is vital for improving patient outcomes. This project aims to develop a predictive model that can identify the onset of sepsis using physiological time-series data.

## Dataset
- **Source**: PhysioNet 2019 Challenge Dataset
- **Contents**: Time-series patient data including vital signs and lab results such as:
  - Heart Rate (HR)
  - Oxygen Saturation (O2Sat)
  - Temperature (Temp)
  - Systolic/Diastolic Blood Pressure (SBP/DBP)
  - Mean Arterial Pressure (MAP)
  - Respiratory Rate (Resp)
  - Demographic and timestamped clinical features
- **Files**: Data extracted from `archive.zip` into `sepsis-physionet` directory

## Methodology

### 1. Data Preprocessing
- Loaded data from training and testing files.
- Combined multiple CSVs and inspected for missing values and data types.
- Imputed missing values using:
  - `SimpleImputer` for median and constant filling
  - `KNNImputer` for leveraging similarity between samples
- Normalized features using `StandardScaler` for uniform feature scaling.

### 2. Exploratory Data Analysis
- Correlation heatmaps to analyze relationships between features
- Violin plots, box plots, and histograms to visualize distribution of HR, O2Sat, MAP, etc.
- Class balance check: Highly imbalanced dataset with many more negative than positive sepsis labels.

### 3. Feature Engineering
- Removed highly correlated features to reduce multicollinearity
- Selected statistically significant variables based on domain knowledge and correlation analysis

### 4. Model Building
- **Logistic Regression**: Baseline classifier
- **Random Forest Classifier**: Ensemble-based model with good interpretability and robustness
- **XGBoost**: Gradient boosting decision trees known for handling imbalance and nonlinear interactions
- **GridSearchCV**: Hyperparameter tuning for model optimization

### 5. Model Evaluation
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
- Visualizations:
  - Confusion matrices
  - ROC curves
- Highlight: XGBoost model achieved higher recall and AUC, indicating stronger performance in detecting sepsis cases.

## Results & Analysis
- **Random Forest** and **XGBoost** both outperformed logistic regression, particularly in recall, which is crucial for sepsis detection (reducing false negatives).
- XGBoost showed strong robustness against missing data and class imbalance.
- Visualizations illustrated how features like heart rate and MAP behaved differently in septic vs non-septic cases.

## Conclusion
This project demonstrates that machine learning models, particularly XGBoost, can effectively predict sepsis using physiological data. Accurate early detection of sepsis can significantly improve patient outcomes and reduce mortality rates. This pipeline provides a foundation for further development into a real-time clinical decision support system.

## Future Work
- Incorporate temporal modeling using RNNs or LSTMs to capture time dependencies.
- Apply SMOTE or ADASYN for class balancing.
- Deploy as a web-based tool for real-time prediction in hospital settings.
- Conduct prospective validation on clinical datasets.

## Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## File Structure
```
├── archive.zip
├── sepsis_prediction.ipynb
├── sepsis-physionet/
│   └── [Extracted CSV files]
```

## Acknowledgements
- PhysioNet 2019 Challenge Team for dataset
- Scikit-learn and XGBoost for ML libraries

## Author
Pushpendra Singh

---
For academic and clinical research inquiries, feel free to reach out via GitHub or email.

