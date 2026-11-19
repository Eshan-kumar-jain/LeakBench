Our loan-default model in `/app/pipeline` reports a validation ROC-AUC of about 0.82, but on new loan applications it performs much worse. Audit the pipeline and fix it so that the validation score it reports is an honest estimate of how the model performs on new applications, and so that the model itself performs as well as it reasonably can.

The data is in `/app/data/loans.csv`; `/app/data/README.md` describes the columns. New applications come from the same population of employers, including some employers we have never lent to before. When a new application is scored, its outcome is of course unknown.

Keep the existing interface:

- `/app/pipeline/features.py` must define `build_features(train_df, apply_df)`. `train_df` is labelled history (it has the `defaulted` column); `apply_df` is the rows to score. It must return one row of model-ready features per row of `apply_df`, with the same index, and it must work when `apply_df` has no `defaulted` column.
- Running `cd /app/pipeline && python train.py` must write:
  - `/app/artifacts/model.joblib`: a fitted model whose `predict_proba` accepts `build_features(loans, new_rows)`, where `loans` is all of `/app/data/loans.csv`.
  - `/app/artifacts/metrics.json` with keys `val_auc` (float) and `features` (list of feature column names).

Run `train.py` yourself so the artifacts exist when you finish. Do not edit anything under `/app/data`.
