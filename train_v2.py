import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

FEATURES = ['rms', 'peak', 'crest_factor', 'energy_50hz', 'energy_100hz', 'energy_150hz']

# --- Load data ---
df = pd.read_csv('real_data_v2.csv')
print(f"Total samples: {len(df)}")
print(f"Healthy: {len(df[df.label == 0])}")
print(f"Faulty:  {len(df[df.label == 1])}")

X = df[FEATURES].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# --- RandomForest ---
print("\nTraining RandomForestClassifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"RF Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
print(classification_report(y_test, y_pred, target_names=['healthy', 'faulty']))

joblib.dump(rf, 'motor_model_v2.pkl')
print("Saved motor_model_v2.pkl")

# --- StandardScaler (fit on all training features) ---
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'scaler_v2.pkl')
print("Saved scaler_v2.pkl")

# --- Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim=6, bottleneck=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Train AE on healthy training samples only
healthy_mask_train = y_train == 0
X_train_healthy = X_train[healthy_mask_train]
X_train_healthy_scaled = scaler.transform(X_train_healthy)

healthy_tensor = torch.FloatTensor(X_train_healthy_scaled)
train_loader = DataLoader(TensorDataset(healthy_tensor), batch_size=8, shuffle=True)

ae = Autoencoder()
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("\nTraining Autoencoder on healthy training samples...")
for epoch in range(200):
    ae.train()
    batch_losses = []
    for (batch,) in train_loader:
        optimizer.zero_grad()
        out = ae(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    if (epoch + 1) % 40 == 0:
        print(f"  Epoch {epoch + 1}/200 | Train loss: {np.mean(batch_losses):.4f}")

# Compute threshold: mean + 2*std on healthy training errors
ae.eval()
with torch.no_grad():
    recon = ae(healthy_tensor)
    errors = ((recon - healthy_tensor) ** 2).mean(dim=1).numpy()

threshold = float(errors.mean() + 3 * errors.std())
print(f"\nAE threshold (mean + 2*std on healthy train): {threshold:.6f}")

joblib.dump(threshold, 'autoencoder_threshold_v2.pkl')
torch.save(ae.state_dict(), 'autoencoder_v2.pth')
print("Saved autoencoder_v2.pth")
print("Saved autoencoder_threshold_v2.pkl")

# --- AE accuracy on test set ---
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_scaled)

with torch.no_grad():
    recon_test = ae(X_test_tensor)
    test_errors = ((recon_test - X_test_tensor) ** 2).mean(dim=1).numpy()

ae_preds = (test_errors > threshold).astype(int)
print(f"\nAE Test Accuracy: {accuracy_score(y_test, ae_preds) * 100:.1f}%")
print(classification_report(y_test, ae_preds, target_names=['healthy', 'faulty']))
