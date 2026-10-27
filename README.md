# GRU Proxy for Optimal Power Flow in an Autonomous Microgrid

A Gated Recurrent Unit (GRU) network, trained in PyTorch, that acts as a fast proxy for a mixed-integer nonlinear (MINLP) optimal power flow (OPF) solver on a 3-bus microgrid with renewables and battery storage. It maps a 24-hour history of system inputs to the optimal dispatch for the next hour. Lehigh University course project (Fall 2025).

**Write-up:** [`docs/Final_Report.pdf`](docs/Final_Report.pdf)

## Problem

Microgrids with wind, solar, and batteries need OPF to minimize operating cost while keeping the system stable. Unit commitment (generators switching on and off), ramp-rate limits, and battery state-of-charge dynamics make the problem mixed-integer, nonlinear, and time-dependent. Traditional solvers are accurate but too slow for real-time control.

**Idea:** solve the OPF offline many times, then train a neural network on the solutions so that inference is a single fast forward pass. A static feedforward network sees only one time step and cannot represent history-dependent constraints, so a recurrent model is used instead.

## System

A closed-loop (mesh) 3-bus microgrid:

| Bus | Resources |
|-----|-----------|
| 1 | CHP generator, Wind turbine 1, PV array 1 |
| 2 | Natural gas generator, Battery 1, PV array 2 (50% of load) |
| 3 | Diesel generator, Battery 2, Wind turbine 2 (50% of load) |

The OPF objective minimizes quadratic generator fuel cost, renewable/battery operating cost, and startup/shutdown penalties. Constraints include AC power balance, unit commitment (binary on/off), generator ramp rates, and battery energy limits.

## Approach

- **Dataset:** 1,680 hourly samples. Ground-truth optimal dispatch comes from an OPF solver (MATPOWER/OPTI) run on load profiles perturbed with noise. Stored in `data/opf_data.mat` as `X_data` (60 input features per hour) and `Y_data` (35 outputs).
- **Inputs:** load demand (active and reactive), solar irradiance, wind speed, and electricity price, as a sequence of the previous 24 hours.
- **Outputs (35):** generator active power and on/off/startup/shutdown states, wind and PV power used, battery power and stored energy, voltage angles and magnitudes, and reactive power for each unit.
- **Preprocessing:** z-score normalization of inputs and outputs, then 24-step sliding windows.
- **Model:** stacked GRU (`nn.GRU`) followed by a linear layer on the last hidden state. Trained with Adam, MSE loss, gradient clipping, and a learning-rate scheduler.
- **Baseline:** a feedforward network (64-64-32, ReLU) taking a single time step, written from scratch in MATLAB with manual backpropagation and mini-batch gradient descent.

## Results

**GRU vs. feedforward baseline.** The baseline failed to learn unit commitment, predicting averaged values instead of on/off states, and had large errors on the natural gas unit's reactive power. The GRU tracked the on/off cycles of the diesel and natural gas units and the battery charge/discharge timing. See the report for the full comparison.

**Per-variable test error (denormalized).**

- Strongest: battery 1 reactive power (about 4×10⁻⁸) and PV 1 reactive power (about 4×10⁻⁴); PV active power follows the daily solar cycle closely.
- Weakest: natural gas reactive power (0.86), wind 2 power used (0.31), CHP active power (0.18), and natural gas active power (0.17). The model tends to underestimate peak natural gas output.
- Battery energy predictions drift from the ground truth over long stretches, consistent with accumulated error in the recurrent state. Bus voltage predictions follow the trend but are noisier than the ground truth.

**Case studies (3 exploratory experiments).**

| Experiment | Settings | Test MSE (normalized) | Train time |
|---|---|---|---|
| Training data size | 20% / 50% / 80% | 1.04 / 0.77 / 0.89 | 70 s / 173 s / 266 s |
| GRU depth | 1 / 3 / 5 layers | 1.28 / 0.70 / 0.88 | 63 s / 258 s / 433 s |
| Convergence | 1000 epochs | final training MSE about 0.1 | 312 s |

Three layers gave the best result in the depth study, and error did not fall monotonically with more data, which suggests sensitivity to noisy samples in a small dataset.

## Notes and limitations

- The dataset is small (1,680 samples), so these results show feasibility rather than a production-ready solver.
- The case studies are exploratory. They split windows randomly, and because the 24-step windows overlap, test windows share data with training windows, so those errors are likely optimistic. `train_opf.py` uses a chronological 80/20 split.
- In the depth study, the 1-layer model also used a smaller hidden size (16 vs. 64), so depth and width are not isolated.
- Predictions are not constrained to satisfy the physical equations. A physics-informed loss (Kirchhoff's-law penalty terms) is the main next step, along with larger networks (IEEE 33-bus and 123-bus systems) and noise-robust training.

## Repository structure

```
.
├── README.md
├── data/
│   └── opf_data.mat
├── python/
│   ├── train_opf.py          # train/load the GRU, evaluate, save plots
│   ├── run_experiments.py    # data size, depth, and convergence case studies
│   └── GRU_architecture.py
├── matlab/
│   └── train_dnn.m           # feedforward baseline (from scratch, no toolboxes)
├── models/
│   └── opf_gru_model.pth
├── results/
│   └── case_study_results/
└── docs/
    └── Final_Report.pdf
```

## Requirements

- Python 3.9+ with `torch`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`
- MATLAB (base install) for the baseline

## How to run

The scripts load `opf_data.mat` from the current folder, so run them from a directory that contains it (or edit the path).

1. **GRU:** `python train_opf.py`. Set `TRAIN_MODEL = True` to train, or `False` to load `opf_gru_model.pth`. The model settings at the top of the script must match the saved weights. Plots are written to `saved_plots/`.
2. **Case studies:** `python run_experiments.py`. Results and timings are written to `case_study_results/`.
3. **Baseline:** run `train_dnn.m` in MATLAB to train and evaluate the feedforward network.

## Author

Alexis Guzman, B.S. Computer Engineering, Lehigh University
[LinkedIn](https://www.linkedin.com/in/alexis-guzman-64a01b245)