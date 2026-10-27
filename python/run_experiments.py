import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import time

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
PLOT_DIR = 'case_study_results'
os.makedirs(PLOT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32          
LEARNING_RATE = 0.001
EPOCHS = 800           
SEQUENCE_LENGTH = 24     
HIDDEN_DIM = 256         
NUM_LAYERS = 3           
DROPOUT = 0.2          
PATIENCE = 50

# ==========================================
# 2. DATA LOADING
# ==========================================
print("Loading Data...")
try:
    mat = scipy.io.loadmat('opf_data.mat')
    X_raw = mat['X_data']
    Y_raw = mat['Y_data']
except FileNotFoundError:
    raise FileNotFoundError("opf_data.mat not found. Please run the MATLAB script first.")

# Normalize
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X_raw)
Y_scaled = scaler_y.fit_transform(Y_raw)

# Create Sequences
xs, ys = [], []
for i in range(len(X_scaled) - SEQUENCE_LENGTH):
    xs.append(X_scaled[i : i+SEQUENCE_LENGTH])
    ys.append(Y_scaled[i+SEQUENCE_LENGTH])
X_seq = np.array(xs)
Y_seq = np.array(ys)

input_dim = X_seq.shape[2]
output_dim = Y_seq.shape[1]

# ==========================================
# 3. MODEL CLASS
# ==========================================
class OPF_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(OPF_GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Define GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # Define Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # Forward pass
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 4. TRAINING FUNCTION (ROBUST)
# ==========================================
def train_model(train_split_pct, num_layers, hidden_dim, epochs):
    # --- STEP 1: GLOBAL RANDOM SPLIT ---
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_seq, Y_seq, test_size=0.2, random_state=42, shuffle=True
    )
    
    # --- STEP 2: SUBSAMPLE TRAINING DATA ---
    num_samples = int(train_split_pct * len(X_train_full))
    indices = np.random.choice(len(X_train_full), num_samples, replace=False)
    
    X_subset = torch.Tensor(X_train_full[indices])
    y_subset = torch.Tensor(y_train_full[indices])
    
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test)
    
    train_loader = DataLoader(TensorDataset(X_subset, y_subset), batch_size=BATCH_SIZE, shuffle=True)
    
    # --- STEP 3: INITIALIZE MODEL ---
    # FIXED: Using the passed 'hidden_dim' argument instead of the global variable
    model = OPF_GRU(input_dim, hidden_dim, output_dim, num_layers, DROPOUT).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)
    criterion = nn.MSELoss()
    loss_curve = []
    
    # --- STEP 4: TRAINING LOOP ---
    start_time = time.time()
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        loss_curve.append(avg_loss)
        scheduler.step(avg_loss)
        
    training_time = time.time() - start_time
    
    # --- STEP 5: EVALUATE ON TEST SET ---
    model.eval()
    with torch.no_grad():
        out_test = model(X_test_tensor.to(device))
        test_mse = criterion(out_test, y_test_tensor.to(device)).item()
        
    return test_mse, training_time, loss_curve

# ==========================================
# 5. RUN EXPERIMENTS
# ==========================================

# Prepare to save times
time_log_path = f"{PLOT_DIR}/training_times.txt"
log_content = []

# --- EXPERIMENT 1: DATA SIZE ---
print("\n--- Running Case Study 1: Data Size ---")
percentages = [0.2, 0.5, 0.8]
mse_results_1 = []
times_1 = []

log_content.append("=== Case Study 1: Impact of Data Size ===")

for p in percentages:
    print(f"Training with {int(p*100)}% of data...")
    mse, duration, _ = train_model(train_split_pct=p, num_layers=3, hidden_dim=64, epochs=EPOCHS)
    mse_results_1.append(mse)
    times_1.append(duration)
    
    msg = f"  {int(p*100)}% Data: {duration:.2f} seconds (MSE: {mse:.4f})"
    print(msg)
    log_content.append(msg)

plt.figure(figsize=(8,6))
bars = plt.bar([f"{int(p*100)}%" for p in percentages], mse_results_1, color=['#ff9999', '#66b3ff', '#99ff99'])
plt.title("Case Study 1: Impact of Training Data Size")
plt.ylabel("Test MSE")
plt.xlabel("Percentage of Data Used")
plt.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    yval = bar.get_height()
    label = f"MSE: {yval:.4f}\n{times_1[i]:.1f}s"
    plt.text(bar.get_x() + bar.get_width()/2, yval, label, ha='center', va='bottom')
plt.savefig(f"{PLOT_DIR}/Case1_DataSize.png")
print("Saved Case1_DataSize.png")


# --- EXPERIMENT 2: NETWORK DEPTH ---
print("\n--- Running Case Study 2: Hidden Layers ---")
layers = [1, 3, 5]
mse_results_2 = []
times_2 = []

log_content.append("\n=== Case Study 2: Impact of Network Depth ===")

for l in layers:
    print(f"Training with {l} Hidden Layers...")
    
    dim = 16 if l == 1 else 64
    
    mse, duration, _ = train_model(train_split_pct=0.8, num_layers=l, hidden_dim=dim, epochs=EPOCHS)
    mse_results_2.append(mse)
    times_2.append(duration)
    
    msg = f"  {l} Layers: {duration:.2f} seconds (MSE: {mse:.4f})"
    print(msg)
    log_content.append(msg)

plt.figure(figsize=(8,6))
plt.plot(layers, mse_results_2, marker='o', linewidth=2, linestyle='--', color='purple')
plt.title("Case Study 2: Impact of Network Depth")
plt.ylabel("Test MSE")
plt.xlabel("Number of GRU Layers")
plt.xticks(layers)
plt.grid(True)

# Annotate with Time
for i, txt in enumerate(times_2):
    plt.annotate(f"{txt:.1f}s", (layers[i], mse_results_2[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig(f"{PLOT_DIR}/Case2_Layers.png")
print("Saved Case2_Layers.png")


# --- EXPERIMENT 3: TRAINING CONVERGENCE ---
print("\n--- Running Case Study 3: Training Epochs ---")
mse, duration, loss_curve = train_model(train_split_pct=0.8, num_layers=3, hidden_dim=64, epochs=1000)

msg = f"  1000 Epochs: {duration:.2f} seconds (Final Loss: {loss_curve[-1]:.4f})"
print(msg)

log_content.append("\n=== Case Study 3: Convergence ===")
log_content.append(msg)

plt.figure(figsize=(8,6))
plt.plot(loss_curve, color='black', linewidth=1.5)
plt.title(f"Case Study 3: Training Convergence (Total Time: {duration:.1f}s)")
plt.ylabel("Training Loss (MSE)")
plt.xlabel("Epochs")
plt.yscale('log')
plt.grid(True, which="both", alpha=0.3)
plt.savefig(f"{PLOT_DIR}/Case3_Epochs.png")
print("Saved Case3_Epochs.png")

# --- SAVE LOG FILE ---
with open(time_log_path, "w") as f:
    f.write("\n".join(log_content))

print(f"\nAll Case Studies Complete. Timing results saved to: {time_log_path}")