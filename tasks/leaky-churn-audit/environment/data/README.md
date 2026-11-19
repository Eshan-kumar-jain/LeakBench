# Data dictionary: `train.csv`

One row per customer per month, months 1 to 12. Customers who churn have no rows after the month they leave.

| Column | Meaning |
| --- | --- |
| `customer_id` | Customer identifier |
| `month` | Calendar month index (1 = first month of data) |
| `plan` | `basic` or `premium` |
| `tenure_months` | Months the customer had been subscribed at the start of this month |
| `monthly_spend` | Amount billed this month |
| `support_calls` | Support calls logged this month |
| `account_status` | CRM status field. The retention team updates it in a review held after each quarter (months 1-3, 4-6, ...) has ended |
| `churned_next_month` | Target: 1 if the customer cancels after this month |

In production the model scores every active customer at the end of each month, using only the data available at that moment.
