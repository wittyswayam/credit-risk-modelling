# Dataset description

**Name:** Default of Credit Card Clients (UCI ML Repository, id 350)
**Origin:** Yeh & Lien (2009), a Taiwanese card issuer, April to September 2005.
**Size:** 30,000 clients, 23 predictors, 1 binary target, 1 ID column. No missing values.
**Target:** `default payment next month` (1 = default in October 2005, 0 = no default). Prevalence 22.12% (6,636 / 30,000).
**Provenance note:** the file was obtained from the UCI download and supplied to this project as a zip
(SHA-256 in `data/README.md`). It could not be re-fetched inside the build sandbox, so the checksum
was not compared against an independent UCI-published hash.

| Variable | Meaning | Notes |
|---|---|---|
| ID | Row identifier | Not a feature; dropped |
| LIMIT_BAL | Credit limit (NT$), individual and family supplementary credit | Range up to 1,000,000 |
| SEX | 1 = male, 2 = female | Candidate for exclusion on fair-lending grounds |
| EDUCATION | 1 = graduate school, 2 = university, 3 = high school, 4 = others | Undocumented codes 0, 5, 6 present (345 rows) |
| MARRIAGE | 1 = married, 2 = single, 3 = others | Undocumented code 0 present (54 rows) |
| AGE | Age in years | 21 to 79 |
| PAY_0, PAY_2 to PAY_6 | Repayment status for Sep, Aug, Jul, Jun, May, Apr 2005 | -1 = paid duly, 1..9 = months of delay. Values -2 and 0 are undocumented. No `PAY_1`: `PAY_0` is September |
| BILL_AMT1 to BILL_AMT6 | Bill statement amount, Sep back to Apr (NT$) | Negative values exist (credit balances) |
| PAY_AMT1 to PAY_AMT6 | Amount paid, Sep back to Apr (NT$) | Non-negative |

## Quality findings
- 0 missing values, 0 fully duplicated rows. 35 rows duplicate another row on every column except ID; they are retained (plausible coincidence among low-activity accounts) and flagged.
- EDUCATION codes 0/5/6 and MARRIAGE 0 are undocumented; the plan is to group them with "others" and say so.
- PAY_x: -2 and 0 are not in the official description. Commonly read as "no consumption" and "revolving credit", but this is an interpretation, not a documented fact.
- PAY_2 to PAY_6 contain almost no value 1 (28, 4, 2, 0, 0 rows) although PAY_0 has 3,688. This coding irregularity is worth reporting, not correcting.
- Negative BILL_AMT values appear in roughly 600 to 700 rows per month.

## Leakage assessment
All predictors are measured April to September 2005; the target is October 2005. There is no target-derived column. The temporal order is valid for a PD model, with caveats: `PAY_0` is the most recent status and is very strongly predictive (default rate 69% at PAY_0 = 2 versus 13% at 0), which is genuine signal rather than leakage. ID order is not documented as chronological, so the split will be random and stratified.
