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

# ==========================================
# 1. CONFIGURATION
# ==========================================
TRAIN_MODEL = False       # Set to FALSE to skip training and load existing model
SAVE_PLOTS = True        # Set to TRUE to save graphs to disk
MODEL_PATH = 'opf_gru_model.pth'
PLOT_DIR = 'saved_plots'

BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 1000
SEQUENCE_LENGTH = 24     
HIDDEN_DIM = 256
NUM_LAYERS = 5
DROPOUT = 0.1
PATIENCE = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if SAVE_PLOTS:
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(PLOT_DIR)}")

OUTPUT_LABELS = [
    "P_chp", "U_chp", "SdnChp", "SupChp", "P_ds", "U_ds", "SdnDs", "SupDs",
    "P_ng", "U_ng", "SdnNg", "SupNg", "Pw1_used", "Pw2_used", "Pv1_used", "Pv2_used",
    "P_b1", "P_b2", "E_b1", "E_b2", "del1", "del2", "del3", "V1", "V2", "V3",
    "Q_chp", "Q_ds", "Q_ng", "Q_w1", "Q_w2", "Q_pv1", "Q_pv2", "Q_b1", "Q_b2"
]

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_and_process_data():
    print("Loading data...")
    try:
        mat = scipy.io.loadmat('opf_data.mat')
        X_raw = mat['X_data'] 
        Y_raw = mat['Y_data'] 
    except FileNotFoundError:
        raise FileNotFoundError("opf_data.mat not found.")

    # Standard Scaler (Fit on RAW data)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit and transform
    X_scaled = scaler_x.fit_transform(X_raw)
    Y_scaled = scaler_y.fit_transform(Y_raw)

    # Create Sequences
    xs, ys = [], []
    for i in range(len(X_scaled) - SEQUENCE_LENGTH):
        x = X_scaled[i : i+SEQUENCE_LENGTH]
        y = Y_scaled[i+SEQUENCE_LENGTH]
        xs.append(x)
        ys.append(y)
    
    X_seq = np.array(xs)
    Y_seq = np.array(ys)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_seq, Y_seq, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler_y, X_seq.shape[2], Y_seq.shape[1]

X_train, X_test, y_train, y_test, scaler_y, input_dim, output_dim = load_and_process_data()

train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)), shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test)), shuffle=False, batch_size=BATCH_SIZE)

# ==========================================
# 3. MODEL DEFINITION
# ==========================================
class OPF_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(OPF_GRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_DIM).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = OPF_GRU(input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS, DROPOUT).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=PATIENCE)

# ==========================================
# 4. TRAIN OR LOAD
# ==========================================
if TRAIN_MODEL:
    print(f"Starting Training ({EPOCHS} epochs)...")
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)
        
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}')

    # Save the model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, linewidth=2)
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True)
    if SAVE_PLOTS: plt.savefig(f"{PLOT_DIR}/training_loss.png", dpi=300)
    plt.close() # Close to free memory

else:
    # LOAD MODE
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Error: Saved model not found! Change TRAIN_MODEL = True.")
        exit()

# ==========================================
# 5. EVALUATION
# ==========================================
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())

pred_norm = np.concatenate(all_preds, axis=0)
true_norm = np.concatenate(all_targets, axis=0)

# Denormalize
Y_pred = scaler_y.inverse_transform(pred_norm)
Y_true = scaler_y.inverse_transform(true_norm)

# MSE Table
mse_per_output = np.mean((Y_true - Y_pred)**2, axis=0)
mse_df = pd.DataFrame({'Output Variable': OUTPUT_LABELS, 'Test MSE': mse_per_output})
print(mse_df.to_string())

# ==========================================
# 6. SAVING PLOTS
# ==========================================
# 1. Set Global Plot Styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'], # Professional fonts
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# 2. Helper to get clean titles and units
def get_plot_metadata(raw_label):
    """Returns (Title, Y_Unit) based on variable name"""
    if raw_label.startswith("P"):
        return f"{raw_label} Active Power", "Power (MW)"
    elif raw_label.startswith("Q"):
        return f"{raw_label} Reactive Power", "Reactive (MVar)"
    elif raw_label.startswith("V"):
        return f"Bus {raw_label} Voltage", "Voltage (kV)"
    elif raw_label.startswith("E_"):
        return f"{raw_label} Stored Energy", "Energy (MWh)"
    elif "del" in raw_label:
        return f"Voltage Angle {raw_label}", "Angle (rad)"
    else:
        return raw_label, "Value"

# 3. Main Plotting Function
def save_and_plot_professional(indices, group_title, subset_slice, filename_tag):
    if subset_slice is None:
        y_t, y_p = Y_true, Y_pred
        x_axis = np.arange(len(Y_true))
        suffix = "Full Test Set"
    else:
        y_t, y_p = Y_true[subset_slice], Y_pred[subset_slice]
        x_axis = np.arange(len(y_t))
        suffix = "Zoomed View"

    num_plots = len(indices)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
    
    if num_plots == 1: axes = [axes]
    
    # Main Figure Title
    fig.suptitle(f"{group_title} - {suffix}", fontweight='bold', y=0.99)
    
    for i, ax in enumerate(axes):
        idx = indices[i]
        label_raw = OUTPUT_LABELS[idx]
        title_text, y_unit = get_plot_metadata(label_raw)
        
        # Plotting Reference (Truth) and Prediction
        # Using Black/Grey for Truth and Red/Blue for Pred is standard in papers
        ax.plot(x_axis, y_t[:, idx], color='#333333', linestyle='-', linewidth=1.5, label='Ground Truth')
        ax.plot(x_axis, y_p[:, idx], color='#d62728', linestyle='--', linewidth=1.5, label='GRU Prediction')
        
        # Labels and Titles
        ax.set_title(title_text, fontweight='bold', pad=10)
        ax.set_ylabel(y_unit)
        
        # Only put legend on the first plot to avoid clutter
        if i == 0:
            ax.legend(loc='upper right', frameon=True, framealpha=0.9)
            
    # X-Label only on the bottom plot
    axes[-1].set_xlabel("Time Step (Hours)", fontweight='bold')
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        clean_suffix = suffix.replace(" ", "_")
        fname = f"{PLOT_DIR}/{filename_tag}_{clean_suffix}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved {fname}")
    
    plt.close(fig)

# 4. Reactive Power Grid Function (Professional)
def plot_grid_reactive_pro(subset_slice, suffix_name):
    y_t = Y_true if subset_slice is None else Y_true[subset_slice]
    y_p = Y_pred if subset_slice is None else Y_pred[subset_slice]
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f"Reactive Power (Q) Analysis - {suffix_name}", fontweight='bold', fontsize=16)
    axes = axes.flatten()
    
    idx_q = list(range(26, 35))
    
    for i, ax in enumerate(axes):
        idx = idx_q[i]
        label_raw = OUTPUT_LABELS[idx]
        title_text, y_unit = get_plot_metadata(label_raw)
        
        ax.plot(y_t[:, idx], color='#333333', label='Truth')
        ax.plot(y_p[:, idx], color='#d62728', linestyle='--', label='Pred')
        
        ax.set_title(title_text, fontsize=10)
        
        if i % 3 == 0:
            ax.set_ylabel(y_unit)
        
        if i >= 6:
            ax.set_xlabel("Time (Hours)")
            
        if i == 0:
            ax.legend(loc='upper right')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if SAVE_PLOTS:
        fname = f"{PLOT_DIR}/Reactive_Power_{suffix_name.replace(' ', '_')}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved {fname}")
    plt.close(fig)

# ==========================================
# 7. EXECUTE PLOTTING
# ==========================================
# Define Groups
idx_gen = [0, 4, 8]        # Generators
idx_wind = [12, 13]         # Wind
idx_pv = [14, 15]           # PV
idx_batt = [16, 17, 18, 19] # Batteries
idx_volt = [23, 24, 25]     # Voltages

full_slice = None
zoom_slice = slice(0, 168) # Zoom first week (168 hours)

# Generate & Save
print("Generating professional plots...")
save_and_plot_professional(idx_gen, "Generator Active Power", full_slice, "Generators")
save_and_plot_professional(idx_gen, "Generator Active Power", zoom_slice, "Generators")

save_and_plot_professional(idx_wind, "Wind Power Dispatch", full_slice, "Wind_Power")
save_and_plot_professional(idx_wind, "Wind Power Dispatch", zoom_slice, "Wind_Power")

save_and_plot_professional(idx_pv, "PV Power Dispatch", full_slice, "PV_Power")
save_and_plot_professional(idx_pv, "PV Power Dispatch", zoom_slice, "PV_Power")

save_and_plot_professional(idx_batt, "Battery Storage Dynamics", full_slice, "Batteries")
save_and_plot_professional(idx_batt, "Battery Storage Dynamics", zoom_slice, "Batteries")

save_and_plot_professional(idx_volt, "Bus Voltage Profiles", full_slice, "Voltages")
save_and_plot_professional(idx_volt, "Bus Voltage Profiles", zoom_slice, "Voltages")

plot_grid_reactive_pro(full_slice, "Full Set")
plot_grid_reactive_pro(zoom_slice, "Zoomed View")

print("Done! All images saved in 'saved_plots' folder.")