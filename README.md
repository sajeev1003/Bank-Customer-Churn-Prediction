# Bank Customer Churn Prediction

This personal machine learning portfolio project predicts bank-customer churn and translates model performance into a practical retention decision:

> Which customers are most likely to churn, what signals drive that risk, and how should the bank convert risk scores into an operational retention strategy?

The analysis moves from data-integrity controls and exploratory statistics to leakage-safe model selection, held-out evaluation, feature interpretation, and recall-aware threshold design. The notebook is fully executed, so every table, chart, metric, and conclusion can be reviewed without rerunning the code.

## Project files

| File | Purpose |
|---|---|
| `Bank_Customer_Churn_Prediction.ipynb` | Complete executable analysis with all code, tables, charts, interpretations, and stored outputs. |
| `Bank_Customer_Churn_Prediction_Report.html` | Browser-friendly rendering of the executed notebook for easy reading and sharing. |
| `Bank Customer Churn Prediction.csv` | Original input dataset used by the verified analysis. |
| `README.md` | Project overview, results, methods, limitations, and rerun instructions. |

## Data and analytical population

- Source data: **10,000 customers and 12 fields**, including the binary `churn` outcome.
- Outcome prevalence: **2,037 customers churned**, giving an overall churn rate of **20.37%**.
- Data-quality checks found **no missing cells, no duplicate rows, no unexpected columns, and 10,000 unique customer IDs**.
- Geographic composition: **5,014 customers from France**, **2,509 from Germany**, and **2,477 from Spain**.
- Validation design: a stratified **80/20 split** produced 8,000 training customers and an untouched 2,000-customer test set. Test churn prevalence is **20.35%**.
- `customer_id` is excluded from modelling because it is a unique account identifier rather than a defensible behavioural predictor.

The outcome is imbalanced enough that accuracy alone is misleading: a model predicting that every customer stays would achieve approximately 79.7% test accuracy while identifying no churners.

## Analytical methods

The project combines complementary methods so that no single metric or chart carries the conclusion:

1. **Data integrity and descriptive analysis**
   - schema validation, type and range checks, missingness review, duplicate detection, and unique-ID verification;
   - churn prevalence, geographic composition, and outcome-based credit-score comparison;
   - confidence intervals and effect sizes to distinguish statistical differences from practically useful separation.

2. **Leakage-safe preprocessing**
   - the train/test split is completed before preprocessing;
   - numerical imputation and scaling, binary passthrough, and categorical imputation and one-hot encoding are contained inside scikit-learn pipelines;
   - all preprocessing parameters are learned from the applicable training fold only.

3. **Machine-learning comparison**
   - Logistic Regression as a transparent linear baseline;
   - Random Forest for nonlinear interactions and bagged-tree robustness;
   - XGBoost for regularized gradient boosting on structured tabular data;
   - five-fold stratified cross-validation with hyperparameter selection based on average precision (PR-AUC).

4. **Held-out validation and decision design**
   - accuracy, precision, recall, F1, PR-AUC, ROC-AUC, ranking curves, and confusion matrices on the untouched test set;
   - an operational threshold selected by maximizing F2 on training out-of-fold predictions, without using test labels;
   - explicit comparison between the conventional 0.50 threshold and a recall-first decision rule.

5. **Model interpretation**
   - held-out permutation importance measured by the decrease in test PR-AUC after feature shuffling;
   - business-readable age, product-count, and membership-activity risk summaries;
   - confidence intervals and sample-size warnings for smaller segments.

## Headline results

- Overall churn is **20.37%**: 2,037 of 10,000 customers left the bank.
- Churned customers have a mean credit score only **6.50 credit-score units lower** than customers who stayed. Cohen's d is approximately **-0.067**, indicating heavy distributional overlap and little practical separation from credit score alone.
- XGBoost leads training-only model selection with **0.699 +/- 0.020 cross-validated PR-AUC**, ahead of Random Forest at **0.684 +/- 0.023** and Logistic Regression at **0.469 +/- 0.031**.
- On the untouched test set, XGBoost reaches **0.716 PR-AUC** and **0.867 ROC-AUC**, making it the strongest risk-ranking model across thresholds.
- At the conventional 0.50 threshold, Random Forest narrowly leads the threshold-specific metrics with **84.8% accuracy**, **61.4% precision**, **67.3% recall**, and **64.2% F1**. XGBoost remains close at **84.5% accuracy**, **60.9% precision**, **66.6% recall**, and **63.6% F1**.
- The training-derived XGBoost threshold of **0.271** raises held-out recall from **66.58% to 85.26%**. It identifies **347 of 407 churners** and reduces false negatives from 136 to 60, at the cost of **39.52% precision** and 531 false-positive contacts.
- Held-out permutation importance identifies **age** and **number of products** as the two strongest ranking signals, followed by membership activity, balance, and country.
- Observed churn rises sharply after age 40: **7.52%** for ages 18-30, **12.09%** for 31-40, **33.97%** for 41-50, and **44.65%** for customers aged 51+.
- Product count has a strongly nonlinear relationship with churn: **27.71%** for one product, **7.58%** for two, **82.71%** for three, and **100%** for four. The four-product result is based on only 60 customers and should not be generalized without more evidence.
- Inactive members churn at **26.85%**, compared with **14.27%** for active members.

These are descriptive associations and predictive results, not causal estimates. They identify where investigation and controlled retention testing should begin.

## Recommended operating sequence

1. **Rank:** use tuned XGBoost probabilities to order customers by estimated churn risk.
2. **Choose capacity:** set the operating threshold from retention value, contact cost, and available campaign capacity. Use the training-derived 0.271 threshold as a recall-first starting point rather than a universal optimum.
3. **Investigate:** prioritize service research around older customers, customers with one or three-plus products, inactive members, high-balance customers, and Germany-based customers.
4. **Experiment:** test retention offers through randomized controlled pilots so the bank learns which actions prevent churn rather than merely predicting it.
5. **Monitor:** track calibration, precision, recall, false-positive cost, subgroup performance, and data drift after deployment.

Age, gender, and country require legal, fairness, and policy review before being used for customer treatment. They can instead be retained for model-performance auditing where appropriate.

## How to rerun the notebook

1. Keep `Bank_Customer_Churn_Prediction.ipynb` and `Bank Customer Churn Prediction.csv` in the same folder.
2. Open the notebook in Google Colab or Jupyter.
3. Select **Run all**.
4. If the CSV is not found locally, the notebook will prompt for upload when running in Colab.

The environment cell checks for required packages and installs only missing dependencies. The verified run uses random seed **42** and completed in approximately **10.48 minutes** with one model-search worker. Runtime will vary by hardware and available parallelism.

The recorded environment was Python 3.12.13, pandas 3.0.5, scikit-learn 1.9.0, and XGBoost 3.4.0. Core packages also include NumPy, Matplotlib, Seaborn, and SciPy. Minor last-decimal differences may occur under different library versions.

## Interpretation guardrails

- The dataset is a single historical snapshot with no time variable, so a random holdout cannot reproduce future-period drift.
- The model estimates churn risk, not whether a specific retention treatment will prevent churn.
- The recall-first threshold is a business trade-off: it catches more churners but creates substantially more false-positive contacts.
- Product-count groups with three and four products are small (`n=266` and `n=60`), so their extreme churn rates carry greater uncertainty.
- Permutation importance measures predictive reliance, not causal impact or a guaranteed direction of effect.
- Country and demographic signals can reflect unmeasured service, lifecycle, economic, or policy differences.
- Before production, the model should be validated on a later cohort, probability-calibrated, fairness-audited, cost-tested, drift-monitored, and evaluated through controlled uplift experiments.

The highest-value next step is to connect customer risk scores with contact cost, offer cost, expected customer value, and observed treatment response, then optimize retention decisions for incremental value rather than churn probability alone.
