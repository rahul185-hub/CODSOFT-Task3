import joblib
import os

model_path = r"C:\Users\rahul\OneDrive\Desktop\customer\model.pkl"
scaler_path = r"C:\Users\rahul\OneDrive\Desktop\customer\scaler.pkl"

print(os.path.exists(model_path))
print(os.path.exists(scaler_path))

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Model expects:", model.n_features_in_)
print("Scaler expects:", scaler.n_features_in_)