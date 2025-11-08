import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

# TensorFlow GPU/CPU info
gpus = tf.config.list_physical_devices('GPU')
device_msg = f"Running on device: {'GPU' if len(gpus) > 0 else 'CPU'}"
print(device_msg)

# ---------- Config ----------
bin_width = 0.1
num_bins = int(1.0 / bin_width)
binwidth_scalar = bin_width  # w for EM calculation

# OptionDataset equivalent (keeps similar API semantics)
class OptionDataset:
    def __init__(self, X, y):
        # Accept pandas DataFrame/Series or numpy arrays
        if isinstance(X, pd.DataFrame):
            self.X = X.values.astype(np.float32)
        else:
            self.X = np.array(X, dtype=np.float32)
        if isinstance(y, pd.Series):
            self.y = y.values.astype(np.int64)
        else:
            self.y = np.array(y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def as_tf_dataset(self, batch_size=32, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.X))
        ds = ds.batch(batch_size)
        return ds

# Residual Block implemented with Keras
class ResidualBlockLayer(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_sizes, **kwargs):
        super(ResidualBlockLayer, self).__init__(**kwargs)
        # Conv1D in Keras expects (batch, steps, channels). We'll keep "steps" = feature length.
        self.conv1 = layers.Conv1D(filters=out_channels, kernel_size=kernel_sizes[0], padding='same', activation='relu')
        self.conv2 = layers.Conv1D(filters=out_channels, kernel_size=kernel_sizes[1], padding='same', activation='relu')
        self.conv3 = layers.Conv1D(filters=out_channels, kernel_size=kernel_sizes[2], padding='same', activation=None)
        # Matching channels with 1x1 conv if required
        if in_channels != out_channels:
            self.match_channels = layers.Conv1D(filters=out_channels, kernel_size=1, padding='same', activation=None)
        else:
            self.match_channels = lambda x: x  # identity

        self.out_act = layers.Activation('relu')

    def call(self, inputs, training=False):
        # inputs shape: (batch, steps, channels)
        residual = self.match_channels(inputs)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual
        return self.out_act(x)

# ResNetClassifier with Keras Functional subclassing
class ResNetClassifierTF(keras.Model):
    def __init__(self, in_channels, num_classes):
        super(ResNetClassifierTF, self).__init__()
        # Note: in_channels corresponds to input channels (we will expand dims so in_channels=1)
        self.blockA = ResidualBlockLayer(in_channels, 64, [3, 2, 1])
        self.blockB = ResidualBlockLayer(64, 128, [3, 2, 1])
        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(num_classes, activation=None)  # logits

    def call(self, inputs, training=False):
        # inputs shape: (batch, features)
        # Expand last axis to create channels dimension -> (batch, steps, channels)
        x = tf.expand_dims(inputs, axis=-1)  # (B, F, 1) -> steps=F, channels=1
        x = self.blockA(x, training=training)
        x = self.blockB(x, training=training)
        x = self.global_pool(x)  # (B, filters)
        return self.fc(x)  # logits

# Option Pricing Formulas (unchanged)
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

# Simulated Data Generator (unchanged)
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

# ----------------- Data selection (same choices as original) -----------------
choice = input("Choose data type:\n1: Real Market Data\n2: Simulated Data\nEnter 1 or 2: ").strip()

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
    tf.random.set_seed(42)
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

# ----------------- Create TF datasets -----------------
batch_size = 128 if choice == '2' else 32
train_dataset_wrapper = OptionDataset(X_train, y_train)
test_dataset_wrapper = OptionDataset(X_test, y_test)
train_ds = train_dataset_wrapper.as_tf_dataset(batch_size=batch_size, shuffle=True)
test_ds = test_dataset_wrapper.as_tf_dataset(batch_size=batch_size, shuffle=False)

# ----------------- Model init and training -----------------
# in_channels = 1 (we expand dims to create channel axis)
model = ResNetClassifierTF(in_channels=1, num_classes=num_bins)

# Build the model by providing an input shape
# Input shape: (None, features)
if isinstance(X_train, pd.DataFrame):
    input_dim = X_train.shape[1]
else:
    input_dim = X_train.shape[1]
# Build by calling once
model.build(input_shape=(None, input_dim))

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

epochs = 100

# Optional: callback to print loss at every 10 epochs (Keras prints per epoch anyway)
class PrintEveryTen(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            loss = logs.get('loss')
            acc = logs.get('accuracy')
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}, Accuracy: {acc:.6f}")

# Fit model
history = model.fit(train_ds, epochs=epochs, verbose=2, callbacks=[PrintEveryTen()])

# ----------------- Inference and metrics -----------------
# Predict on test set
# We'll collect inputs and true labels similarly
all_inputs = []
all_true = []
all_preds = []

for batch_X, batch_y in test_ds:
    preds_logits = model(batch_X, training=False).numpy()
    preds = np.argmax(preds_logits, axis=1)
    all_preds.extend(preds.tolist())
    all_true.extend(batch_y.numpy().tolist())
    all_inputs.extend(batch_X.numpy().tolist())

# Save predictions to CSV
output_df = pd.DataFrame(all_inputs, columns=X.columns if choice == '1' else ["S/K", "1", "T", "r", "sigma"])
output_df['True_Bin'] = all_true
output_df['Predicted_Bin'] = all_preds
output_filename = "output_predictions.csv" if choice == '1' else "predictions_output.csv"
output_df.to_csv(output_filename, index=False)
print(f"Saved predictions to: {output_filename}")

# ----------------- Error Metric (EM) and Inaccuracy Metric (RHO) -----------------
true_arr = np.array(all_true)
pred_arr = np.array(all_preds)
Tlen = len(true_arr)
em = (binwidth_scalar / Tlen) * np.sum(np.abs(true_arr - pred_arr))
rho = np.sum(np.abs(true_arr - pred_arr) > 2) / Tlen

print(f"\n--- Evaluation Metrics ---")
print(f"Error Metric (EM): {em:.6f}")
print(f"Inaccuracy Metric (RHO): {rho:.6f}")

# ----------------- Line plot -----------------
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
