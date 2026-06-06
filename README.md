# Advanced Machine Learning Model Validation ⚙️

A robust, production-ready Machine Learning pipeline featuring automated data preprocessing, hyperparameter optimization, and strict model evaluation. This repository demonstrates a complete workflow using `scikit-learn` and `XGBoost` to solve classification tasks, ensuring reproducibility via unified pipelines and cross-validation techniques.

## 📋 Project Overview
Building reliable machine learning systems requires more than just fitting a model; it demands data leakage prevention, systematic tuning, and exhaustive performance diagnostics. This project implements an enterprise-grade pipeline that automates missing data imputation, feature scaling, model selection (comparing Ensemble techniques vs. Gradient Boosting), and advanced statistical evaluation (ROC-AUC curves and multi-metric matrices).

## ✨ Core Pipeline Architecture

### 1. Unified Preprocessing Pipeline (`ColumnTransformer`)
To eliminate data leakage and ensure clean data transformation, the workflow encapsulates preprocessing steps within a `scikit-learn` pipeline:
* **Missing Value Imputation:** Utilizes `SimpleImputer` to handle missing entities safely across feature arrays.
* **Feature Scaling:** Implements `StandardScaler` to normalize numerical data, establishing zero mean and unit variance for optimum algorithmic convergence.
* **Component Assembly:** Combines distinct feature transformations seamlessly using `ColumnTransformer` prior to model execution.

### 2. Model Exploration & Advanced Algorithms
The pipeline evaluates and benchmarks two of the most powerful algorithms for structured data:
* **Random Forest (`RandomForestClassifier`):** An ensemble bagging classifier utilized to establish strong baseline metrics and measure feature importance variances.
* **XGBoost (`XGBClassifier`):** An optimized gradient boosting framework designed for high-performance, speed, and superior predictive accuracy.

### 3. Hyperparameter Tuning & Cross-Validation
* **Grid Search Optimization (`GridSearchCV`):** Automates hyperparameter search across multi-dimensional grids to isolate the best-performing model configurations.
* **Cross-Validation (`cross_val_score`):** Implements $k$-fold cross-validation to assess model stability, mitigate overfitting, and guarantee generalization on unseen data distributions.
* **Data Splitting:** Employs stratified `train_test_split` execution to maintain consistent class proportions across training and testing subsets.

### 4. Rigorous Evaluation Metrics & Diagnostics
Models are evaluated beyond simple accuracy using a multi-metric approach:
* **Classification Performance:** Simultaneous extraction of `accuracy_score`, `precision_score`, `recall_score`, and `f1_score`.
* **Error Mapping:** Generates a structured `confusion_matrix` visualized via `Seaborn` to analyze True/False Positives and Negatives.
* **Receiver Operating Characteristic (`roc_curve`, `auc`):** Computes and plots ROC curves alongside Area Under the Curve (AUC) scores to gauge the classifier's discriminative power at various thresholds.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Data Core:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Pipelines, Compose, Model Selection)
* **Gradient Boosting:** XGBoost
* **Data Visualization:** Matplotlib, Seaborn

## 📁 Repository Structure
```text
├── models/                     # Serialized pipelines and optimized model weights
├── validation_pipeline.py      # Main machine learning pipeline execution script
├── environment.yml             # Conda dependency configuration file
└── README.md                   # Project documentation
```
🚀 Getting Started
1. Clone the Repository:
   Bash
   git clone [https://github.com/longaresf/advanced-ml-model-validation.git](https://github.com/longaresf/advanced-ml-model-validation.git)
   cd advanced-ml-model-validation

2. Execute the Validation Framework:
   Ensure your Python environment is active with the required dependencies (scikit-learn, xgboost, pandas, seaborn), then run:

   Bash
   python validation_pipeline.py

✒️ Autor

   Francisco Longares - Científico de Datos & Desarrollador de Soluciones IA - longaresf
