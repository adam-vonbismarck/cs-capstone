#!/usr/bin/env python3
import os
import glob
import re
import argparse
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import ruptures as rpt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Constants for oar geometry
INBOARD_MM = 1135.0
OAR_LENGTH_MM = 3690.0
OUTBOARD_MM = OAR_LENGTH_MM - INBOARD_MM
INBOARD_M = INBOARD_MM / 1000.0
OUTBOARD_M = OUTBOARD_MM / 1000.0
OAR_LENGTH_M = OAR_LENGTH_MM / 1000.0
SAMPLE_FREQ = 50.0
DELTA_T = 1.0 / SAMPLE_FREQ

# Savitzky-Golay defaults
SG_WINDOW = 51
SG_POLY = 3

class StrokeDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)


def denoise_df(df, cols):
    """
    Apply Savitzky-Golay filter to each column in cols.
    """
    return pd.DataFrame({col: savgol_filter(df[col].values, SG_WINDOW, SG_POLY) for col in cols})


def detect_cp_indicator(signal, pen=None):
    """
    Use PELT change-point detection on 1D signal.
    Returns a binary indicator array where odd segments are marked as strokes (1).
    """
    n = len(signal)
    algo = rpt.Pelt(model="l2").fit(signal)
    if pen is None:
        pen = np.std(signal) * 3
    cps = algo.predict(pen=pen)
    indicator = np.zeros(n, dtype=int)
    start = 0
    for i, end in enumerate(cps):
        val = 1 if i % 2 == 1 else 0
        indicator[start:end] = val
        start = end
    return indicator


def compute_session_power(df, force_cols, angle_cols, vel_cols):
    """
    Compute average instantaneous power per rower over full session,
    using the same approach as in chkpts.py.
    """
    avg_powers = []
    for i, fcol in enumerate(force_cols):
        # Convert raw force (kgf) to Newtons
        forces = df[fcol].values * 9.80665
        angles = np.deg2rad(df[angle_cols[i]].values)

        angular_velocity = np.deg2rad(df[vel_cols[i]].values)

        handle_speed = INBOARD_M * angular_velocity * np.cos(angles)

        handle_force = forces * (OUTBOARD_M / OAR_LENGTH_M)
        positive_idx = handle_speed > 0

        if np.any(positive_idx):
            power_inst = handle_force[positive_idx] * handle_speed[positive_idx]
            avg_power = np.mean(power_inst)
        else:
            avg_power = 0.0

        avg_powers.append(avg_power)

    return avg_powers


def extract_features(df, force_cols, angle_cols, vel_cols):
    """
    Extract summary statistics and stroke ratio features and targets per rower.
    """
    # Denoise signals
    f_dn = denoise_df(df, force_cols)
    a_dn = denoise_df(df, angle_cols)
    v_dn = denoise_df(df, vel_cols)

    # Compute average watts
    targets = compute_session_power(df, force_cols, angle_cols, vel_cols)
    # Single stroke indicator from GateForceX_8, reused for all rowers
    stroke_ind = detect_cp_indicator(f_dn[force_cols[-1]].values)
    stroke_ratio = stroke_ind.mean()
    features = []
    for i, fcol in enumerate(force_cols):
        feats = []
        feats.extend([f_dn[fcol].mean(), f_dn[fcol].std()])
        feats.extend([a_dn[angle_cols[i]].mean(), a_dn[angle_cols[i]].std()])
        feats.extend([v_dn[vel_cols[i]].mean(), v_dn[vel_cols[i]].std()])
        feats.append(stroke_ratio)  # fraction of time in stroke
        features.append(feats)
    return np.array(features), np.array(targets)


def main(data_dir, output_folder):
    # Gather all session files
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(files)} session files.")
    all_feats = []
    all_targets = []
    # Loop sessions
    for fpath in files:
        print(f"Processing {fpath}")
        df = pd.read_csv(fpath, encoding='utf-16', delimiter='\t', header=[0,1], engine='python').iloc[:,:-1]
        df.columns = ['_'.join(col).strip() if col[1] else col[0].strip() for col in df.columns.values]
        df.columns = ['Time'] + list(df.columns[1:])
        df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
        if df['Time'].max() > 1e6:
            df['Time'] = df['Time'] / 1000.0
        # Select cols
        force_cols = sorted([c for c in df.columns if re.match(r"^GateForceX_\d+$", c)])[:8]
        angle_cols = sorted([c for c in df.columns if re.match(r"^GateAngle_\d+$", c) and 'Vel' not in c])[:8]
        vel_cols = sorted([c for c in df.columns if re.match(r"^GateAngleVel_\d+$", c)])[:8]
        if not force_cols or not angle_cols or not vel_cols:
            print(f"Skipping {fpath}: missing gate columns")
            continue
        feats, targets = extract_features(df, force_cols, angle_cols, vel_cols)
        all_feats.append(feats)
        all_targets.append(targets)
    # Flatten rows and rowers
    X = np.vstack(all_feats)
    y = np.hstack(all_targets)
    # Train/test split
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    train_ds = StrokeDataset(X[train_idx], y[train_idx])
    test_ds = StrokeDataset(X[test_idx], y[test_idx])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)
    # Define models
    model1 = MLP(input_dim=6)  # without stroke_ratio
    model2 = MLP(input_dim=7)  # with stroke_ratio
    criterion = nn.MSELoss()
    optim1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    optim2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    # Training loop
    print("Training...")
    def train(mdl, optim):
        mdl.train()
        for epoch in range(50):
            print(f"Epoch {epoch + 1}/{50}")
            for xb, yb in train_loader:
                inp = xb[:,:6] if mdl is model1 else xb
                pred = mdl(inp)
                loss = criterion(pred, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()
    train(model1, optim1)
    train(model2, optim2)
    # Evaluation
    def eval_model(mdl):
        mdl.eval()
        errors = []
        with torch.no_grad():
            for xb, yb in test_loader:
                inp = xb[:,:6] if mdl is model1 else xb
                pred = mdl(inp)
                errors.append(((pred - yb) ** 2).mean().item())
        return np.mean(errors)
    mse1 = eval_model(model1)
    mse2 = eval_model(model2)
    print(f"Baseline Model (no stroke flag) MSE: {mse1:.4f}")
    print(f"Augmented Model (with stroke flag) MSE: {mse2:.4f}")
    # Save models
    os.makedirs(output_folder, exist_ok=True)
    torch.save(model1.state_dict(), os.path.join(output_folder, 'model1.pth'))
    torch.save(model2.state_dict(), os.path.join(output_folder, 'model2.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train power prediction models.")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV files.")
    parser.add_argument("--output", default="model_output", help="Output folder for models.")
    args = parser.parse_args()
    main(args.data_dir, args.output)
