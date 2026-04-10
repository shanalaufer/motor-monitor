import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load real data
df = pd.read_csv('real_data.csv', header=None)
df.columns = ['rms', 'peak', 'crest_factor', 'energy_50hz', 'energy_100hz', 'energy_150hz', 'label']

print(f"Total samples: {len(df)}")
print(f"Healthy: {len(df[df.label==0])}")
print(f"Faulty: {len(df[df.label==1])}")

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.1f}%")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion matrix:")
print(f"                 Predicted healthy  Predicted faulty")
print(f"Actually healthy       {cm[0,0]}                {cm[0,1]}")
print(f"Actually faulty        {cm[1,0]}                {cm[1,1]}")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=['healthy', 'faulty']))

print("\nFeature importance:")
importances = model.feature_importances_
for feature, importance in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
    bar = '█' * int(importance * 50)
    print(f"  {feature:<20} {bar} {importance:.3f}")

joblib.dump(model, 'motor_model.pkl')
print("\nReal data model saved as motor_model.pkl")