import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# Load real data
df = pd.read_csv('real_data.csv', header=None)
df.columns = ['rms', 'peak', 'crest_factor', 'energy_50hz', 'energy_100hz', 'energy_150hz', 'label']

print(f"Total samples: {len(df)}")
print(f"Healthy: {len(df[df.label==0])}")
print(f"Faulty:  {len(df[df.label==1])}")

# Split healthy and faulty
healthy = df[df.label==0].drop('label', axis=1).values
faulty  = df[df.label==1].drop('label', axis=1).values

# Normalize — fit scaler on healthy data only
scaler = StandardScaler()
healthy_scaled = scaler.fit_transform(healthy)
faulty_scaled  = scaler.transform(faulty)

# Save scaler for later use in receiver.py
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

# Convert to tensors
healthy_tensor = torch.FloatTensor(healthy_scaled)

# Train/val split on healthy data
split = int(0.8 * len(healthy_tensor))
train_data = healthy_tensor[:split]
val_data   = healthy_tensor[split:]

train_loader = DataLoader(TensorDataset(train_data), batch_size=8, shuffle=True)

# Define autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim=6, bottleneck=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, bottleneck),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Train
print("\nTraining autoencoder...")
train_losses = []
val_losses = []

for epoch in range(200):
    model.train()
    batch_losses = []
    for (batch,) in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_output = model(val_data)
        val_loss = loss_fn(val_output, val_data).item()
        val_losses.append(val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/200 | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

# Plot training curve
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Autoencoder training curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_curve.png')
plt.show()
print("Training curve saved as training_curve.png")

# Compute reconstruction errors
model.eval()
with torch.no_grad():
    healthy_tensor_all = torch.FloatTensor(healthy_scaled)
    faulty_tensor_all  = torch.FloatTensor(faulty_scaled)

    healthy_errors = loss_fn(model(healthy_tensor_all), healthy_tensor_all).item()
    faulty_errors_list = []
    for i in range(len(faulty_tensor_all)):
        sample = faulty_tensor_all[i].unsqueeze(0)
        err = loss_fn(model(sample), sample).item()
        faulty_errors_list.append(err)

    healthy_errors_list = []
    for i in range(len(healthy_tensor_all)):
        sample = healthy_tensor_all[i].unsqueeze(0)
        err = loss_fn(model(sample), sample).item()
        healthy_errors_list.append(err)

print(f"\nMean reconstruction error:")
print(f"  Healthy: {np.mean(healthy_errors_list):.4f}")
print(f"  Faulty:  {np.mean(faulty_errors_list):.4f}")
print(f"  Ratio:   {np.mean(faulty_errors_list)/np.mean(healthy_errors_list):.1f}x higher for faulty")

# Find optimal threshold
all_errors = healthy_errors_list + faulty_errors_list
all_labels = [0]*len(healthy_errors_list) + [1]*len(faulty_errors_list)

best_threshold = None
best_f1 = 0
for threshold in np.percentile(healthy_errors_list, range(50, 100)):
    preds = [1 if e > threshold else 0 for e in all_errors]
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nOptimal threshold: {best_threshold:.4f}")
predictions = [1 if e > best_threshold else 0 for e in all_errors]
print("\nClassification report:")
print(classification_report(all_labels, predictions, target_names=['healthy', 'faulty']))

cm = confusion_matrix(all_labels, predictions)
print(f"Confusion matrix:")
print(f"                 Predicted healthy  Predicted faulty")
print(f"Actually healthy       {cm[0,0]}                {cm[0,1]}")
print(f"Actually faulty        {cm[1,0]}                {cm[1,1]}")

# Plot reconstruction errors
plt.figure(figsize=(10, 4))
plt.scatter(range(len(healthy_errors_list)), healthy_errors_list,
            label='Healthy', color='steelblue', alpha=0.7, s=30)
plt.scatter(range(len(healthy_errors_list), len(all_errors)), faulty_errors_list,
            label='Faulty', color='crimson', alpha=0.7, s=30)
plt.axhline(y=best_threshold, color='orange', linestyle='--',
            label=f'Threshold: {best_threshold:.4f}')
plt.xlabel('Sample')
plt.ylabel('Reconstruction error')
plt.title('Autoencoder reconstruction error — healthy vs faulty')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('reconstruction_errors.png')
plt.show()
print("Error plot saved as reconstruction_errors.png")

# Save model and threshold
torch.save(model.state_dict(), 'autoencoder.pth')
joblib.dump(best_threshold, 'autoencoder_threshold.pkl')
print(f"\nModel saved as autoencoder.pth")
print(f"Threshold saved as autoencoder_threshold.pkl")