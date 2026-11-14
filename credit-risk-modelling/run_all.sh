#!/usr/bin/env bash
# Execute all notebooks in order, in place. Requires data/default of credit card clients.xls.
set -euo pipefail
cd "$(dirname "$0")/notebooks"
for nb in 01_exploratory_data_analysis 02_data_preprocessing 03_logistic_regression_modelling 04_credit_risk_analysis; do
  echo "Running $nb"
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb"
done
echo "Done. Figures in results/figures, tables in results/, models in models/."
