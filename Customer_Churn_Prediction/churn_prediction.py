import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid")

# 1. LOAD
df = pd.read_csv(os.path.join(BASE_DIR, "telco.csv"))
print("Shape:", df.shape)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)   # new customers with tenure 0
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print("Churn rate: %.1f%%" % (df["Churn"].mean() * 100))

# 2. EXPLORE
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for a, col in zip(ax.flat[:4], ["Contract", "InternetService", "PaymentMethod", "TechSupport"]):
    (df.groupby(col)["Churn"].mean() * 100).sort_values().plot.barh(ax=a, color="#0057ff")
    a.set_title(f"Churn % by {col}")
sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, ax=ax[1, 1], common_norm=False)
ax[1, 1].set_title("Tenure distribution")
sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, ax=ax[1, 2], common_norm=False)
ax[1, 2].set_title("Monthly charges distribution")
plt.tight_layout(); plt.savefig("eda_churn_patterns.png", dpi=130); plt.close()

# 3. FEATURE ENGINEERING
svc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
       "TechSupport", "StreamingTV", "StreamingMovies"]
df["num_services"] = (df[svc] == "Yes").sum(axis=1)
df["has_support"] = ((df["TechSupport"] == "Yes") | (df["OnlineSecurity"] == "Yes")).astype(int)
df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
df["tenure_group"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                            labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
df["auto_pay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
df["charge_vs_avg"] = df["MonthlyCharges"] - df["avg_monthly_spend"]

X = pd.get_dummies(df.drop(columns=["customerID", "Churn"]), drop_first=True)
y = df["Churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges",
            "num_services", "avg_monthly_spend", "charge_vs_avg"]
sc = StandardScaler().fit(X_tr[num_cols])
X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
X_tr_s[num_cols] = sc.transform(X_tr[num_cols])
X_te_s[num_cols] = sc.transform(X_te[num_cols])

# 4. TRAIN
models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"), True),
    "Random Forest": (RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                      class_weight="balanced", random_state=42, n_jobs=-1), False),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), False),
}
results, fitted = [], {}
for name, (m, scaled) in models.items():
    Xa, Xb = (X_tr_s, X_te_s) if scaled else (X_tr, X_te)
    m.fit(Xa, y_tr)
    fitted[name] = (m, Xb)
    p, pr = m.predict(Xb), m.predict_proba(Xb)[:, 1]
    rep = classification_report(y_te, p, output_dict=True)
    cv = cross_val_score(m, Xa, y_tr, cv=5, scoring="roc_auc").mean()
    results.append(dict(Model=name, Accuracy=rep["accuracy"], Precision=rep["1"]["precision"],
                        Recall=rep["1"]["recall"], F1=rep["1"]["f1-score"],
                        ROC_AUC=roc_auc_score(y_te, pr), CV_AUC=cv))
res = pd.DataFrame(results).round(3)
print("\n", res.to_string(index=False))
res.to_csv("model_comparison.csv", index=False)

# 5. EVALUATE
best = res.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
m, Xb = fitted[best]
print("\nBest model:", best)
print(classification_report(y_te, m.predict(Xb), target_names=["Retained", "Churned"]))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_te, m.predict(Xb)),
                       display_labels=["Retained", "Churned"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title(f"Confusion matrix - {best}")
for n, (mm, xx) in fitted.items():
    RocCurveDisplay.from_estimator(mm, xx, y_te, ax=ax[1], name=n)
ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set_title("ROC curves")
plt.tight_layout(); plt.savefig("evaluation.png", dpi=130); plt.close()

# Feature importance
pi = permutation_importance(m, Xb, y_te, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(12)
print("\nTop churn drivers:\n", imp.round(4))
imp[::-1].plot.barh(figsize=(8, 6), color="#0057ff")
plt.title("Top 12 churn drivers"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=130); plt.close()

# At-risk customers
df_te = df.loc[X_te.index].copy()
df_te["churn_prob"] = m.predict_proba(Xb)[:, 1]
top = df_te.sort_values("churn_prob", ascending=False).head(10)[
    ["customerID", "Contract", "tenure", "MonthlyCharges", "churn_prob"]]
top.to_csv("top_at_risk_customers.csv", index=False)
print("\nTop 10 at-risk customers:\n", top.round(2).to_string(index=False))