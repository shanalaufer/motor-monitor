import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv('motor_dataset.csv')

# Separate features from labels
X = df.drop('label', axis=1)  # everything except the label column
y = df['label']                # just the label column

# Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# Train the Random Forest
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Done!")

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.1f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion matrix:")
print(f"                 Predicted healthy  Predicted faulty")
print(f"Actually healthy       {cm[0,0]}                {cm[0,1]}")
print(f"Actually faulty        {cm[1,0]}                {cm[1,1]}")

# Full report
print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=['healthy', 'faulty']))

# Feature importance — what did the model actually learn?
print("\nFeature importance (what mattered most):")
importances = model.feature_importances_
for feature, importance in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
    bar = '█' * int(importance * 50)
    print(f"  {feature:<20} {bar} {importance:.3f}")

# Save the model for later use in the dashboard
joblib.dump(model, 'motor_model.pkl')
print("\nModel saved as motor_model.pkl")