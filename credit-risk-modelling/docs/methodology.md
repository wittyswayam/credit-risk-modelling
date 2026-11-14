# Methodology

Status: all analysis stages are executed (see the notebooks). Sections list the design that was fixed before fitting, followed by what was actually found.

## 1. Research questions
1. Can logistic regression predict next-month default on this portfolio?
2. Which borrower characteristics are associated with default probability?
3. How do L1/L2 regularisation and class weighting change performance?
4. Are predicted probabilities calibrated?
5. How should a classification threshold be chosen?
6. What are the consequences of false positives versus false negatives?

## 2. Data
UCI Default of Credit Card Clients (id 350): 30,000 clients, 22.12% default prevalence, no missing values. See `dataset_description.md` for variables, anomalies and the provenance caveat.

## 3. Splitting strategy 
- Stratified random split on the target: **60% train / 20% validation / 20% test**, seed 42.
- Row order is not documented as chronological and all rows share one observation window, so a time-based split is not available. This is a limitation: it cannot test stability across economic periods.
- Train: fitting and cross-validated hyperparameter search (stratified 5-fold). Validation: model comparison, calibration, threshold selection. Test: touched once, for the final report.

## 4. Preprocessing (scikit-learn `Pipeline` + `ColumnTransformer`)
- Merge EDUCATION 0, 4, 5, 6 into "others"; MARRIAGE 0 into 3; one-hot encode.
- PAY_0 to PAY_6 kept as ordinal numeric values; -2/-1/0 not recoded.
- Signed log transform for BILL_AMT and log1p for PAY_AMT and LIMIT_BAL, then standard scaling. Outliers are transformed, not deleted.
- Candidate engineered features (utilisation, payment ratios, age squared) are judged on validation data only.
- SEX excluded from the main model (fair-lending concern) and reported as a sensitivity check.
- Every step is fitted on the training fold only, inside the pipeline, to prevent leakage.

## 5. Models 
Logistic regression only: (1) baseline with minimal regularisation, (2) L1, (3) L2, (4) class-weighted. Regularisation strength tuned by cross-validated ROC-AUC (PR-AUC as a secondary check). No tree or neural models in the main experiments.

## 6. Evaluation 
Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, KS statistic, confusion matrix. Accuracy alone is insufficient because the majority-class baseline already scores 77.9%.

## 7. Financial risk analysis 
PD risk bands (illustrative thresholds), odds ratios with statsmodels significance and VIF, calibration before and after recalibration on validation data, cost-based threshold analysis, and a clearly labelled hypothetical ECL sensitivity analysis. The dataset has no LGD or EAD, so actual expected credit loss cannot be computed.

## 8. EDA findings that shaped the design (executed)
- 22.1% prevalence implies PR-AUC, recall and calibration matter.
- PAY_0 is the dominant signal (default rate 12.8% at status 0, 33.9% at one month late, 69.1% at two).
- Default falls monotonically with credit limit (31.8% to 13.8% across quintiles) and is U-shaped in age.
- BILL_AMT columns are strongly collinear (pairwise correlation at least 0.80), which motivates regularisation and VIF diagnostics.

## 9. Decisions and results (executed)
- Feature set: engineered features, SEX excluded. Validation ROC-AUC 0.7478 versus 0.7306 for base features; SEX changed it by +0.0003.
- Tuning: stratified 5-fold CV ROC-AUC on the training set over C in 10^-4 to 10^2 (13 values). Selected L2, C = 0.01 (rule fixed in advance: best of baseline/L1/L2 on validation ROC-AUC, L2 preferred if within 0.002).
- Calibration: recalibrators were assessed out-of-fold on validation; none improved Brier by more than 0.0003 for the L2 model, so raw probabilities are used.
- Threshold: minimum validation cost with a hypothetical false-negative to false-positive cost ratio of 5, giving 0.13.
- Test set: evaluated once for the selected L2 model: ROC-AUC 0.765, PR-AUC 0.520, KS 0.424, Brier 0.139.
- A class-weighted model with Platt recalibration performs comparably on Brier and slightly better on ROC-AUC on validation (bootstrap interval for the difference 0.0027 to 0.0073); it was not evaluated on the test set.
- The planned split, preprocessing and model families were followed. Deviations: calibration and thresholds were evaluated on out-of-fold validation predictions and the repository gained extra result tables beyond the two CSVs in the original structure.

- Disclosure: EDA (notebook 01) used all rows, and notebook 02 prints row and default counts of the test split (no performance data). Design choices informed by full-data EDA are listed in the README limitations.
