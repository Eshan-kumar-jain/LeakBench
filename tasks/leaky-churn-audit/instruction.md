Our churn model in `/app/pipeline` reports a validation ROC-AUC of about 0.99, but when it scored new months in production it did far worse. Audit the pipeline and fix it so that the validation score it reports is an honest estimate of how the model will perform on future months.

The data is in `/app/data/train.csv` (months 1 to 12); `/app/data/README.md` describes the columns. In production the model scores every active customer at the end of each month, using only the data available at that moment.

Keep the existing interface:

- `/app/pipeline/features.py` must define `build_features(df)`, returning one row of model-ready features per input row with the same index. It must not use the target column.
- Running `cd /app/pipeline && python train.py` must write:
  - `/app/artifacts/model.joblib`: a fitted model whose `predict_proba` accepts the output of `build_features`.
  - `/app/artifacts/metrics.json` with keys `val_auc` (float), `train_months` and `val_months` (lists of the month numbers used for training and validation) and `features` (list of feature column names).

Run `train.py` yourself so the artifacts exist when you finish. You may change the model, features and validation scheme, but do not edit anything under `/app/data`.
