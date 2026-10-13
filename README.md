# Task 3 – Customer Churn Prediction

**Dataset:** Telco Customer Churn (IBM sample data, via Kaggle). It has 7,043 customers and 21 columns, and 26.5% of the customers churned.

**How to run**
```bash
pip install pandas scikit-learn matplotlib seaborn
python churn_prediction.py
```

## Approach
1. **Load & clean:** Fixed 11 blank `TotalCharges` values (new customers with tenure 0).
2. **Explore:** Compared churn rates across contract, internet service, payment method, tech support and tenure (`eda_churn_patterns.png`).
3. **Feature engineering:** Added `num_services`, `has_support`, `avg_monthly_spend`, `tenure_group`, `auto_pay` and `charge_vs_avg`.
4. **Modelling:** Trained Logistic Regression, Random Forest and Gradient Boosting on a stratified 80/20 split.
5. **Evaluation:** Accuracy, precision, recall, F1, ROC-AUC, 5-fold cross-validation and permutation importance.

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | 0.789 | 0.615 | 0.842 |
| Random Forest | 0.772 | 0.556 | 0.703 | 0.621 | 0.841 |
| Gradient Boosting | 0.806 | 0.678 | 0.513 | 0.584 | 0.846 |

All three models reach a ROC-AUC of about 0.84. Gradient Boosting is the most accurate, but Logistic Regression catches the most churners (79% recall). That matters when missing a churner costs more than a wasted retention offer.

## Key findings
- **Contract:** Month-to-month customers churn at 42.7%, against 11.3% on one-year and 2.8% on two-year contracts.
- **Tenure:** Customers in their first 12 months churn at 47.4%, against 17.1% after that.
- **Internet service:** Fiber optic customers churn at 41.9%, against 19.0% on DSL.
- **Payment method:** Electronic-check payers churn at 45.3%, against 15–19% for other methods.
- **Tech support:** Customers without it churn at 41.6%, against 15.2% with it.
- **Top model drivers:** tenure, contract type, fiber optic internet and monthly charges.

## Business recommendations
1. Offer discounts or incentives to move month-to-month customers onto annual contracts.
2. Focus retention efforts on the first 12 months, with onboarding calls and welcome offers.
3. Investigate fiber optic service quality and pricing.
4. Encourage customers to switch from electronic check to automatic payment.
5. Bundle tech support and online security.
6. Use `top_at_risk_customers.csv` to prioritise outreach.

*These are correlations, not proof of cause. Test the ideas with A/B experiments before investing heavily.*

## Files
- `churn_prediction.py`: full pipeline
- `eda_churn_patterns.png`, `evaluation.png`, `feature_importance.png`: charts
- `model_comparison.csv`, `top_at_risk_customers.csv`: outputs