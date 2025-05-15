import re
import torch
import numpy as np
import pandas as pd
from power_prediction import MLP, extract_features

fpath = "data/csv_9_10_1.csv"
df = pd.read_csv(fpath, encoding='utf-16', delimiter='\t', header=[0,1], engine='python').iloc[:,:-1]
df.columns = ['_'.join(col).strip() if col[1] else col[0].strip() for col in df.columns.values]
df.columns = ['Time'] + list(df.columns[1:])
df = df.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
if df['Time'].max() > 1e6:
    df['Time'] /= 1000.0

force_cols = sorted([c for c in df.columns if re.match(r"^GateForceX_\d+$", c)])[:8]
angle_cols = sorted([c for c in df.columns if re.match(r"^GateAngle_\d+$", c) and "Vel" not in c])[:8]
vel_cols   = sorted([c for c in df.columns if re.match(r"^GateAngleVel_\d+$", c)])[:8]

X, y = extract_features(df, force_cols, angle_cols, vel_cols)

device = torch.device("cpu")
m1 = MLP(input_dim=6).to(device)
m1.load_state_dict(torch.load("model_output/model1.pth", map_location=device))
m1.eval()
m2 = MLP(input_dim=7).to(device)
m2.load_state_dict(torch.load("model_output/model2.pth", map_location=device))
m2.eval()

X1 = torch.from_numpy(X[:,:6]).float()
X2 = torch.from_numpy(X).float()
with torch.no_grad():
    p1 = m1(X1).squeeze().numpy()
    p2 = m2(X2).squeeze().numpy()

print("Model1 (no stroke‐flag) MSE:", np.mean((p1 - y)**2))
print("Model2 (with stroke‐flag) MSE:", np.mean((p2 - y)**2))
print("Per-rower predictions:", p2)