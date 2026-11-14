# CV-ready description

**Credit Risk Modelling: Probability of Default with Logistic Regression** (Python, scikit-learn, statsmodels) · github.com/wittyswayam

*Short form (2 lines)*
- Built a leakage-safe logistic regression PD model on 30,000 UCI credit card clients (22% default rate); held-out test ROC-AUC 0.765, KS 0.42, Brier 0.139, with probabilities matching observed default frequencies without recalibration.
- Compared unregularised, L1, L2 and class-weighted models; analysed calibration, cost-based threshold selection and a hypothetical ECL = PD × LGD × EAD sensitivity.

*Long form (bullets)*
- Verified and documented the dataset (undocumented category codes, repayment-status coding anomalies); designed a reproducible scikit-learn Pipeline/ColumnTransformer with a stratified 60/20/20 split and a single-use test set.
- Showed regularisation barely changes discrimination (CV ROC-AUC 0.764 to 0.765) but stabilises coefficients; interpreted odds ratios and VIF diagnostics; found recent and repeated delinquency to be the dominant default signal.
- Evaluated calibration (reliability curves, ECE, Brier), showed class weighting inflates mean PD (43.6% vs 22.1% observed) and that Platt scaling restores it, and framed threshold choice as a cost-ratio decision (recall 28% at 0.50 vs 79% at 0.13).
- Documented limitations honestly: single time window, 2005 Taiwan context, one-month horizon, no LGD/EAD, no fairness audit.
