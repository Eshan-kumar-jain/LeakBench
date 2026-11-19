Our 30-day readmission model in `/app/pipeline` reports a validation ROC-AUC of about 0.74, but when the hospital used it on patients admitted for the first time it did noticeably worse. Audit the pipeline and fix it so that the validation score it reports is an honest estimate of how the model performs on new patients, and so that the model itself performs as well as it reasonably can on them.

The data is in `/app/data/visits.csv`; `/app/data/README.md` describes the columns. Each patient can have several visits. The model will be used on patients the hospital has not seen before, scoring each of their visits in turn.

Keep the existing interface:

- `/app/pipeline/features.py` must define `build_features(df)`, returning one row of model-ready features per visit with the same index. It must work without the `readmitted_30d` column.
- Running `cd /app/pipeline && python train.py` must write:
  - `/app/artifacts/model.joblib`: a fitted model whose `predict_proba` accepts the output of `build_features`.
  - `/app/artifacts/metrics.json` with keys `val_auc` (float), `features` (list of feature names), and `train_patient_ids` and `val_patient_ids` (lists of the patient IDs whose visits were used for training and for validation).

Run `train.py` yourself so the artifacts exist when you finish. Do not edit anything under `/app/data`.
