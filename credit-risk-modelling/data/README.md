# Data

Dataset: **Default of Credit Card Clients** (UCI Machine Learning Repository, id 350).
Source page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

**The data file is not included in this repository.** Its redistribution terms and provenance could not be
verified when the project was packaged, so it is downloaded separately into `data/raw/` (ignored by git).

## Obtain the data (either way)
1. **Script:** from the repository root run `python scripts/download_data.py`.
2. **Manually:** download the zip from the source page above, extract `default of credit card clients.xls`
   into `data/raw/`.

Then check: `python -c "from src.data_loader import load_raw; print(load_raw().shape)"` should print `(30000, 25)`.

## Checksum used in this project (SHA-256 of the zip archive)
`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`
This was computed on a copy supplied manually during development; it was not compared with an independently
published UCI value. Check the licence stated on the UCI page before reusing or redistributing the data.

Original study: Yeh, I-C. & Lien, C-H. (2009), "The comparisons of data mining techniques
for the predictive accuracy of probability of default of credit card clients",
Expert Systems with Applications 36(2). Taiwan, April to September 2005.
