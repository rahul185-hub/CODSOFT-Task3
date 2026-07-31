import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("customer_churn_data.csv")

print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values if any
df["InternetService"] = df["InternetService"].fillna("")

# ==========================
# Select Features
# ==========================

X = df[["Age", "Gender", "Tenure", "MonthlyCharges"]].copy()

# Encode Gender
X["Gender"] = X["Gender"].map({
    "Male": 0,
    "Female": 1
})

# Encode Target
Y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nTarget Distribution:")
print(Y.value_counts())

# ==========================
# Split Dataset
# ==========================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

print("\nScaler saved successfully.")

# ==========================
# SVM + GridSearchCV
# ==========================

svm = SVC(
    class_weight="balanced",
    random_state=42
)

param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

grid = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\nTraining model...")

grid.fit(X_train, Y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

# ==========================
# Save Model
# ==========================

joblib.dump(best_model, "model.pkl")

print("\nModel saved successfully.")

# ==========================
# Predictions
# ==========================

pred = best_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(Y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, pred))

print("\nClassification Report:")
print(classification_report(Y_test, pred))

print("\nUnique Predictions:")
print(np.unique(pred))

# ==========================
# Test Sample Predictions
# ==========================

samples = pd.DataFrame(
    [
        [25, 0, 60, 40],
        [55, 1, 2, 120],
        [35, 0, 36, 55],
        [48, 1, 8, 95]
    ],
    columns=["Age", "Gender", "Tenure", "MonthlyCharges"]
)

samples_scaled = scaler.transform(samples)

predictions = best_model.predict(samples_scaled)

print("\nSample Predictions:")

for i, p in enumerate(predictions, start=1):
    if p == 1:
        print(f"Customer {i}: Churn")
    else:
        print(f"Customer {i}: Not Churn")