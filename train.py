import json
import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel

# Load extracted features
with open("baby_cry_features.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Drop unnecessary columns
df.drop(columns=["Cry_Audio_File", "Cry_Reason"], inplace=True)

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Feature Selection - Keep only important features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
selector = SelectFromModel(rf, threshold="mean", prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train optimized Random Forest model
model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=42)
model.fit(X_train_selected, y_train)

# Make predictions
y_pred = model.predict(X_test_selected)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# Save model and scaler
joblib.dump(model, "baby_cry_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "feature_selector.pkl")

print("Optimized Model saved as 'baby_cry_classifier.pkl'")
print("Scaler saved as 'scaler.pkl'")
print("Feature Selector saved as 'feature_selector.pkl'")
