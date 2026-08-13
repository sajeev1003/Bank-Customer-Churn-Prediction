# Bank Customer Churn Prediction

This is a personal machine learning portfolio project that predicts which bank customers are most likely to leave and translates those predictions into a practical retention strategy.

## Included files

- `Bank_Customer_Churn_Prediction.ipynb` - the primary editable and executable Google Colab/Jupyter notebook. It contains all code, tables, charts, interpretations, and project sections A1-D11.
- `Bank_Customer_Churn_Prediction_Report.html` - a read-only browser version of the executed notebook for convenient review without Python.
- `Bank Customer Churn Prediction.csv` - the dataset used by the verified run.

## How to run

### Google Colab

1. Upload `Bank_Customer_Churn_Prediction.ipynb` to Colab.
2. Run all cells.
3. When prompted, upload `Bank Customer Churn Prediction.csv` if it is not already in `/content`.

### Local Jupyter

1. Keep the notebook and CSV in the same folder.
2. Use Python 3.10 or newer.
3. Run all cells from top to bottom. The first code cell installs only missing packages.

The latest verified single-worker run completed in about 10.5 minutes. Colab can use parallel search workers and may finish faster.

## Verified headline results

- Overall churn rate: 20.37% (2,037 of 10,000 customers).
- Model selected by five-fold cross-validated average precision: XGBoost.
- XGBoost test PR-AUC: 0.716; ROC-AUC: 0.867.
- At threshold 0.50: accuracy 0.845, precision 0.609, recall 0.666, F1 0.636.
- Training-only F2 threshold: 0.271; held-out recall 0.853 (347 of 407 churners found).
- Leading held-out permutation features: age and number of products.

The threshold result is a business trade-off, not a universal optimum: the recall-first rule reduces missed churners but increases false-positive retention contacts.

## Reproducibility

- Random seed: 42
- Verified runtime: Python 3.12.13, pandas 3.0.5, scikit-learn 1.9.0, XGBoost 3.4.0
