# Project Name: Credit Card Fraud Detection

# Step 1: Load dataset and explore structure
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib

data = pd.read_csv("creditcard.csv")

#print(data.head())
#print(data.info())
#print(data.describe())
#print(data['Class'].value_counts())

#Step 2: Preprocessing and Train/Test Split
X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print("Initial training set distribution:")
print(y_train.value_counts())
print(y_test.value_counts())

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Step 3: Train the ML model (Logistic Regression/Random Forest)
ml_model = LogisticRegression(max_iter=1000)

ml_model.fit(X_train, y_train)

y_pred = ml_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


#Step 4: Confusion Matrix + ROC Curve + Classification Report
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

y_scores = ml_model.decision_function(X_test)
roc_auc = roc_auc_score(y_test, y_scores)
print("ROC-AUC:", roc_auc)

RocCurveDisplay.from_predictions(y_test, y_pred)

#Step 5: Class Imbalance Fixed Using SMOTE
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(X_train_resampled, y_train_resampled)
lr_pred = lr_model.predict(X_test)

print("\n=== Logistic Regression ===")
print("F1:", f1_score(y_test, lr_pred))


#print(y_train_resampled.value_counts())



#Step 6: Random Forest Model Trained and Evaluated
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_resampled, y_train_resampled)

rf_pred = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print("F1:", f1_score(y_test, rf_pred))

xgb_model = XGBClassifier (
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb_model.fit(X_train_resampled, y_train_resampled)
xgb_pred = xgb_model.predict(X_test)

print("\n=== XGBoost ===")
print("F1:", f1_score(y_test, xgb_pred))

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)


joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(xgb_model, "xgb_fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nAll models and scaler saved successfully!")
print("Random Forest Accuracy: ", rf_acc)
print("Random Forest Precision:", rf_prec)
print("Random Forest Recall:", rf_rec)
print("Random Forest F1:", rf_f1)


rf_cm = confusion_matrix(y_test, rf_pred)
print("Random Forest Confusion Matrix:")
print(rf_cm)

print("Logistic Regression F1:", f1_score(y_test, lr_pred))
print("Random Forest F1:", rf_f1)

importances = rf_model.feature_importances_
print(importances)

