import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the model, scaler, and feature selector
model = joblib.load("baby_cry_classifier.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

# Load test data (make sure the data is in the same format as used for training)
with open("baby_cry_features.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Drop unnecessary columns
df.drop(columns=["Cry_Audio_File", "Cry_Reason"], inplace=True)

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Normalize features
X_scaled = scaler.transform(X)

# Select important features
X_selected = selector.transform(X_scaled)

# Split dataset into training (80%) and testing (20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
