# Quality assessment

Performed after the final code change, on a clean copy of the repository in a fresh virtual environment (Python 3.12, pandas 3.1, scikit-learn 1.9, statsmodels 0.15), with `results/` and `models/` deleted first.

## Verified by execution
| Check | Result |
|---|---|
| Dataset shape, dtypes, target definition | 30,000 x 25, all integer, binary target, 22.12% default rate (asserted in `load_raw` and in tests) |
| Fresh-environment execution | `pip install -r requirements.txt`, then `bash run_all.sh`: all four notebooks run start to finish, no errors |
| Reproducibility | All 10 result CSVs regenerate with a maximum absolute difference of 0; the saved L2 model gives identical predictions in a different environment |
| Automated tests | `pytest -q`: 9 tests pass (shape and target, category cleaning, stratified disjoint split, scaler fitted on training rows only, feature ranges, SEX excluded, KS on known cases, ECL arithmetic, calibration bounds) |
| Training-only fitting | Notebook 02 asserts the fitted scaler mean equals the training-row mean and differs from the all-rows mean; preprocessing sits inside the `Pipeline`, so cross-validation refits it per fold |
| Test-set discipline | Test-split variables are used for performance only in notebook 04, section 6 onward, once, for the selected model. Notebook 02 prints the split's row and default counts only |
| Charts match outputs | All 25 figures are produced by the notebooks from the same objects as the tables; the README's figure links all resolve |
| README numbers | Headline, validation and test figures were compared with the saved CSVs and agree |
| Repository hygiene | No data file, archive, cache or local absolute path in the repository or notebook outputs; the dataset is excluded by `.gitignore` |

## Known issues and honest caveats
1. **Exploratory analysis used all rows**, including the later validation and test rows, and informed design choices (category merging, age squared, log transform, utilisation cap, age centre). No parameters were fitted on it, but the test set is not strictly unseen by the analyst.
2. **Validation reuse.** The validation set was used for feature-set choice, model comparison, the calibration decision and the threshold. Only the test result is an unbiased estimate, and it comes from a single split of 6,000 rows, with no confidence intervals.
3. **Test above validation.** Test ROC-AUC (0.765) exceeds validation (0.750); this is consistent with sampling noise but was not tested formally.
4. **Class-weighted alternative.** With Platt recalibration it matches L2's Brier score and ranks slightly better on validation. It was excluded by a rule written in advance and not evaluated on test.
5. **Small model differences** among the baseline, L1 and L2 variants were not tested formally. p-values are uncorrected for multiple testing.
6. **Thresholds and ECL** rest on hypothetical cost ratios and LGD/CCF assumptions. They illustrate sensitivity and are not estimates of real loss.
7. **Dataset provenance.** The data file was supplied manually because the build sandbox could not reach UCI. Its checksum was not compared with an independently published UCI value, and the licence (believed CC BY 4.0) was not confirmed against the UCI page.
8. **Not done:** fairness audit, out-of-time validation, external validation, review by a second person.
