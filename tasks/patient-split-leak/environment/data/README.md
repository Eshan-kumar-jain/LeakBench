# Data dictionary: `visits.csv`

One row per hospital visit; 3,000 patients, most with several visits.

| Column | Meaning |
| --- | --- |
| `patient_id` | Hospital patient number |
| `visit_number` | 1 for the patient's first visit, 2 for the second, ... |
| `age` | Age in years |
| `chronic_conditions` | Number of diagnosed chronic conditions |
| `bmi` | Body-mass index |
| `length_of_stay` | Days in hospital on this visit |
| `num_medications` | Medications prescribed on this visit |
| `emergency_admission` | 1 if this visit started in the emergency department |
| `readmitted_30d` | Target: 1 if readmitted within 30 days of discharge |
