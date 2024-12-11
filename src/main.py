import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
# Replace 'health_data.csv' with your dataset file path
data = pd.read_csv('meet/bmi.csv')

# Features and Target
X = data[['age', 'height', 'weight', 'bmi']]  # Feature columns
y = data['bmiClass']  # Target column (e.g., Normal, Overweight, Obese Class)

# Encode target variable if it's categorical
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importance")
plt.show()

import joblib

# Save the trained model
joblib.dump(rf_model, 'rf_model.pkl')
import joblib
from sklearn.preprocessing import LabelEncoder

# Assume you have already created and fitted the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(['Normal', 'Overweight', 'Obese Class 1', 'Obese Class 2', 'Obese Class 3'])

# Save the label encoder to a file
joblib.dump(label_encoder, 'label_encoder.pkl')