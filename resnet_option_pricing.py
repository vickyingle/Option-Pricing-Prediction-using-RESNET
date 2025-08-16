import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ---------- Config ----------
bin_width = 0.1
num_bins = int(1.0 / bin_width)
binwidth_scalar = bin_width  # w for EM calculation

# Dataset Class
class OptionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32)
        self.y = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[0], padding='same')
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_sizes[1], padding='same')
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_sizes[2], padding='same')
        self.match_channels = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.match_channels(x)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += residual
        return F.relu(out)

# ResNet(2,2) Model
class ResNetClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNetClassifier, self).__init__()
        self.blockA = ResidualBlock(in_channels, 64, [3, 2, 1])
        self.blockB = ResidualBlock(64, 128, [3, 2, 1])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, F)
        x = self.blockA(x)
        x = self.blockB(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

# Option Pricing Formulas
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K + 1e-8) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + 1e-8)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def geo_asian_call_price(S, K, T, r, sigma):
    sigma_hat = sigma / np.sqrt(3)
    mu_hat = 0.5 * (r - 0.5 * sigma ** 2) + sigma_hat ** 2
    d1 = (np.log(S / K + 1e-8) + (mu_hat + 0.5 * sigma_hat ** 2) * T) / (sigma_hat * np.sqrt(T) + 1e-8)
    d2 = d1 - sigma_hat * np.sqrt(T)
    return np.exp(-r * T) * (S * np.exp(mu_hat * T) * norm.cdf(d1) - K * norm.cdf(d2))

# Simulated Data Generator
def generate_data(formula_fn, n_samples=10000):
    S = np.random.uniform(10, 500, n_samples)
    K = np.random.uniform(0.7 * S, 1.3 * S)
    T = np.random.uniform(1/250, 3, n_samples)
    r = np.random.uniform(0.01, 0.03, n_samples)
    sigma = np.random.uniform(0.05, 0.9, n_samples)
    price = formula_fn(S, K, T, r, sigma)

    S_by_K = S / K
    ones = np.ones_like(S)
    features = np.stack([S_by_K, ones, T, r, sigma], axis=1)

    price_ratio = price / K
    bin_indices = np.minimum((price_ratio / bin_width).astype(int), num_bins - 1)
    return features, bin_indices

# Dataset choice
choice = input("Choose data type:\n1: Real Market Data\n2: Simulated Data\nEnter 1 or 2: ")

if choice == '1':
    print("Using Real Market Data...")
    df = pd.read_csv("realMarketData.csv")
    df = df.dropna()
    df = df[
        ['Underlying Value', 'Strike Price', 'T', 'implied_volatility',
         'Prev_Settle', 'Delta_S', 'Turnover_Rate', 'Settle Price']
    ].rename(columns={'Settle Price': 'Target'})
    X = df.drop(columns='Target')
    y_cont = df['Target']
    n_bins = num_bins
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_class = est.fit_transform(y_cont.values.reshape(-1, 1)).astype(int).flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

elif choice == '2':
    print("Using Simulated Data...")
    print("Choose option type to simulate:\n1. Black-Scholes European Option\n2. Geometric Asian Option")
    sim_choice = input("Enter 1 or 2: ").strip()
    if sim_choice == '1':
        formula_fn = bs_call_price
        print("Simulating Black-Scholes European options...")
    elif sim_choice == '2':
        formula_fn = geo_asian_call_price
        print("Simulating Geometric Asian options...")
    else:
        raise ValueError("Invalid option choice.")
    np.random.seed(42)
    torch.manual_seed(42)
    X_np, y_np = generate_data(formula_fn, n_samples=10000)
    df_sim = pd.DataFrame(X_np, columns=["S/K", "1", "T", "r", "sigma"])
    df_sim["Price Bin"] = y_np
    df_sim.to_csv("simulated_data_class.csv", index=False)
    print("Simulated data saved to: simulated_data_class.csv")
    split = int(0.8 * len(X_np))
    X_train, X_test = X_np[:split], X_np[split:]
    y_train, y_test = y_np[:split], y_np[split:]

else:
    raise ValueError("Invalid choice. Enter 1 or 2.")

# Create DataLoaders
batch_size = 128 if choice == '2' else 32
train_dataset = OptionDataset(X_train, y_train)
test_dataset = OptionDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model init and training
model = ResNetClassifier(in_channels=1, num_classes=num_bins).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.6f}")

# Inference and metrics
model.eval()
all_preds, all_true, all_inputs = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y_batch.numpy())
        all_inputs.extend(X_batch.cpu().numpy())

# Save predictions to CSV
output_df = pd.DataFrame(all_inputs, columns=X.columns if choice == '1' else ["S/K", "1", "T", "r", "sigma"])
output_df['True_Bin'] = all_true
output_df['Predicted_Bin'] = all_preds
output_filename = "output_predictions.csv" if choice == '1' else "predictions_output.csv"
output_df.to_csv(output_filename, index=False)
print(f"Saved predictions to: {output_filename}")

# Error Metric (EM) and Inaccuracy Metric (RHO)
true_arr = np.array(all_true)
pred_arr = np.array(all_preds)
T = len(true_arr)
em = (binwidth_scalar / T) * np.sum(np.abs(true_arr - pred_arr))
rho = np.sum(np.abs(true_arr - pred_arr) > 2) / T

print(f"\n--- Evaluation Metrics ---")
print(f"Error Metric (EM): {em:.6f}")
print(f"Inaccuracy Metric (RHO): {rho:.6f}")

# Line plot
plt.figure(figsize=(10, 4))
plt.plot(true_arr[:100], label='True Bin', marker='o')
plt.plot(pred_arr[:100], label='Predicted Bin', marker='x')
plt.title("True vs Predicted Bins (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Bin Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
