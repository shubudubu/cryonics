import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
model = joblib.load("baby_cry_classifier.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")
with open("baby_cry_features.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)
df.drop(columns=["Cry_Audio_File", "Cry_Reason"], inplace=True)
X_train = df.drop(columns=["Label"])
feature_names = X_train.columns.tolist()
def predict_baby_cry(new_cry_features):
    new_data = pd.DataFrame([new_cry_features], columns=feature_names)
    new_data_scaled = scaler.transform(new_data)
    new_data_selected = selector.transform(new_data_scaled)
    prediction = model.predict(new_data_selected)
    cry_types = {0: "belly_pain", 1: "burping", 2: "discomfort", 3: "hungry", 4: "tired"}
    predicted_label = prediction[0]
    predicted_cry_type = cry_types.get(predicted_label, "Unknown")
    return predicted_label, predicted_cry_type
##new_cry_features = {enter the features} 
predicted_label, predicted_cry_type = predict_baby_cry(new_cry_features)
print(f"Predicted Label: {predicted_label}")
print(f"Predicted Crying Type: {predicted_cry_type}")
