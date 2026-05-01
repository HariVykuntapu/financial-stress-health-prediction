"""
Retrain XGBoost model with cleaned feature data.

Fixes vs original training:
1. Filter out BRFSS invalid codes (INCOME2: 77/99, SMOKE100: 7/9, DIABETE3: 7/9)
2. Drop financial_stress_score from model inputs — it is derived from INCOME2/MEDCOST/EMPLOY1,
   so including both creates multicollinearity that breaks SMOTE synthetic samples
   (SMOTE interpolates INCOME2=3.7 while FSS stays calibrated to integer values).
   FSS remains a displayed research metric in the app; the model uses the raw components.
3. Use scale_pos_weight instead of SMOTE — avoids synthetic sample artifacts in
   high-dimensional space where derived and raw features are inconsistent.
4. Reduce max_depth 7→5 to limit overfitting.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/brfss_features.csv"))
print(f"Loaded: {df.shape[0]:,} rows")

df = df[df["INCOME2"].between(1, 8)]
df = df[df["SMOKE100"].isin([1, 2])]
df = df[df["DIABETE3"].isin([1, 2, 3, 4])]
df = df.dropna(subset=["INCOME2", "MEDCOST", "EMPLOY1", "_BMI5", "SMOKE100", "_AGE80", "SEX", "DIABETE3"])
print(f"After cleaning: {df.shape[0]:,} rows")

# financial_stress_score excluded — derived from INCOME2/MEDCOST/EMPLOY1
feature_cols = ["INCOME2", "MEDCOST", "EMPLOY1", "_BMI5", "SMOKE100", "_AGE80", "SEX", "DIABETE3"]

X = df[feature_cols]
y = df["MICHD"].astype(int)
neg, pos = y.value_counts()[0], y.value_counts()[1]
scale_pos_weight = neg / pos
print(f"Target: {neg:,} negative, {pos:,} positive  ->  scale_pos_weight={scale_pos_weight:.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    use_label_encoder=False,
    n_jobs=-1,
)
xgb.fit(X_train, y_train)
print("Model trained.")

y_prob = xgb.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
print(f"AUROC:     {roc_auc_score(y_test, y_prob):.4f}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\n=== Feature importances ===")
for feat, val in sorted(zip(feature_cols, xgb.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {val:.4f}")

joblib.dump(xgb, os.path.join(BASE_DIR, "models/xgb_best_model.pkl"))
joblib.dump(feature_cols, os.path.join(BASE_DIR, "models/feature_cols.pkl"))
print("\nSaved: models/xgb_best_model.pkl, models/feature_cols.pkl")

print("\n=== Sanity check: INCOME2 sweep (medcost=0, employed, age=45, bmi=27, female, non-smoker, no diabetes) ===")
for inc in range(1, 9):
    row = {"INCOME2": inc, "MEDCOST": 0, "EMPLOY1": 1, "_BMI5": 2700,
           "SMOKE100": 2, "_AGE80": 45, "SEX": 1, "DIABETE3": 3}
    X_in = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)
    prob = xgb.predict_proba(X_in)[0][1] * 100
    print(f"  INCOME2={inc}: {prob:.1f}%")

print("\n=== Sanity check: low vs high risk profiles ===")
scenarios = [
    ("Low  (income=8, employ=1, medcost=0, age=35, bmi=22, no-smoke, no-diab)",
     {"INCOME2": 8, "MEDCOST": 0, "EMPLOY1": 1, "_BMI5": 2200, "SMOKE100": 2, "_AGE80": 35, "SEX": 1, "DIABETE3": 3}),
    ("Mid  (income=5, employ=1, medcost=0, age=50, bmi=30, no-smoke, no-diab)",
     {"INCOME2": 5, "MEDCOST": 0, "EMPLOY1": 1, "_BMI5": 3000, "SMOKE100": 2, "_AGE80": 50, "SEX": 2, "DIABETE3": 3}),
    ("High (income=1, employ=2, medcost=1, age=65, bmi=40, smoke, diabetic)",
     {"INCOME2": 1, "MEDCOST": 1, "EMPLOY1": 2, "_BMI5": 4000, "SMOKE100": 1, "_AGE80": 65, "SEX": 2, "DIABETE3": 1}),
]
for label, row in scenarios:
    X_in = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)
    prob = xgb.predict_proba(X_in)[0][1] * 100
    print(f"  {label}: {prob:.1f}%")
