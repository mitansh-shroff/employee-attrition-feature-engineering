# Employee Attrition — Feature Engineering

Feature engineering pipeline for an employee attrition dataset — EDA, imputation, encoding, skew correction, ratio-based feature construction, VIF-based selection, and PCA, wrapped in a reusable scikit-learn pipeline.

## Overview

This project turns a raw, 39-column employee dataset into a clean, model-ready feature matrix for predicting attrition. It walks through the full feature engineering lifecycle:

1. **Understanding the Dataset** — descriptive statistics, missing value checks, feature classification, distribution and outlier analysis, and correlation analysis.
2. **Feature Engineering Strategy** — missing value imputation, categorical encoding (one-hot for nominal fields, ordinal for fields with a genuine order), log1p transforms for skewed and zero-inflated numeric columns, and feature scaling.
3. **Feature Construction & Selection** — new ratio-based features (e.g. compensation relative to experience, promotion velocity, an overtime/work-life balance score) followed by iterative VIF-based elimination to remove redundant features.
4. **Dimensionality Reduction & Pipeline Design** — PCA on correlated satisfaction scores, and a final reusable `scikit-learn` `Pipeline` that fits all transformations (imputers, encoders, scalers, PCA) on training data only.

A full write-up of the reasoning and findings — including the discovery that two supplied columns were exact functions of other columns, caught via VIF — is in [`Summary.pdf`](./Summary.pdf).

## Repository Structure

```
employee-attrition-feature-engineering/
├── README.md
├── .gitignore
├── employee_attrition_feature_engineering.ipynb   # Main analysis & pipeline
├── Employee_Attrition_Dataset.csv                 # Raw dataset
└── Summary.pdf                                    # Written summary & reflection
```

> **Note:** The notebook loads the dataset with a relative path, so the `.csv` file must stay in the same folder as the notebook.

## How to Run

1. Clone the repo:
   ```
   git clone https://github.com/mitansh-shroff/employee-attrition-feature-engineering
   cd employee-attrition-feature-engineering
   ```
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Launch the notebook:
   ```
   jupyter notebook employee_attrition_feature_engineering.ipynb
   ```
4. Run all cells from top to bottom.

## Dataset

The dataset contains 39 employee-level attributes (demographic, compensation, engagement, and performance related) used to analyze and predict employee attrition.

## Author

Mitansh Shroff
