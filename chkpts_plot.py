import pandas as pd
import numpy as np
import ruptures as rpt
import os
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the data efficiently
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-16', delimiter='\t')
    df = df.iloc[1:, :-1].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='any').reset_index(drop=True)
    return df

# Preprocess data: Combine GateForceX and GateAngle columns and average
def preprocess_data(df, interval_seconds=1):
    if df['Time'].max() > 1e6:
        df['Time'] = df['Time'] / 1000.0

    # Identify columns
    gate_force_cols = [col for col in df.columns if 'GateForceX' in col]
    gate_angle_cols = [col for col in df.columns if 'GateAngle' in col and 'Vel' not in col]
    gate_angle_vel_cols = [col for col in df.columns if 'GateAngleVel' in col]

    # Ensure all required columns are present
    required_cols = ['Time', 'Speed'] + gate_force_cols + gate_angle_cols + gate_angle_vel_cols
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} is missing from the data.")

    df = df.dropna(subset=required_cols)

    df = df.sort_values('Time').reset_index(drop=True)

    df['TimeInterval'] = (df['Time'] // interval_seconds).astype(int)  # integer division

    aggregation_dict = {'Time': 'first', 'Speed': 'mean'}
    for col in gate_force_cols + gate_angle_cols + gate_angle_vel_cols:
        aggregation_dict[col] = 'mean'

    df_grouped = df.groupby('TimeInterval').agg(aggregation_dict).reset_index(drop=True)

    return df_grouped

# Normalize the features for change point detection
def normalize_features(df):
    # Normalize the 'Speed' column
    scaler = MinMaxScaler()
    df[['Speed']] = scaler.fit_transform(df[['Speed']])
    return df

# Detect change points for a single variable
def detect_change_points_single(data_series, model="l2", n_bkps=6):
    data = data_series.values.reshape(-1, 1)
    algo = rpt.Binseg(model=model).fit(data)
    change_points = algo.predict(n_bkps=n_bkps)
    # Remove the last point (end of data)
    if change_points and change_points[-1] == len(data_series):
        change_points = change_points[:-1]
    return change_points

# Save change points to a CSV file
def save_change_points(df, change_points, variable_name, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Ensure change points are within bounds
    valid_change_points = [cp for cp in change_points if cp < len(df)]
    if len(valid_change_points) == 0:
        print(f"No valid change points detected for {variable_name}.")
        return
    timestamps = df.iloc[valid_change_points]['Time']
    output_file = os.path.join(output_folder, f"change_points_{variable_name}.csv")
    timestamps.to_csv(output_file, index=False, header=["Time"])
    print(f"Change points for {variable_name} saved to {output_folder}")

def compute_watts_and_length(df, change_points, gate_force_cols, gate_angle_cols, gate_angle_vel_cols):
    watts_list = []
    length_list = []

    num_sections = len(change_points) // 2  # Number of sections between even and odd change points
    print(f"Computing watts and effective length for {num_sections} sections.")

    # Constants from the logger
    inboard = 1135  # mm
    oar_length = 3690  # mm
    outboard = oar_length - inboard  # mm

    for i in range(0, len(change_points) - 1, 2):
        start_idx = change_points[i]
        end_idx = change_points[i + 1] if (i + 1) < len(change_points) else len(df)
        section_df = df.iloc[start_idx:end_idx]

        # Initialize dictionaries for watts and effective lengths
        watts = {}
        lengths = {}

        for idx in range(len(gate_force_cols)):
            force_col = gate_force_cols[idx]
            angle_col = gate_angle_cols[idx]

            # Handle speed and force: convert to proper units
            inboard_m = inboard / 1000.0
            outboard_m = outboard / 1000.0
            oar_length_m = oar_length / 1000.0
            angles_vals = section_df[angle_col].values
            angles_rad = np.deg2rad(angles_vals)
            angular_velocity = np.gradient(angles_rad) * 50  # rad/s
            handle_speed = inboard_m * angular_velocity * np.cos(angles_rad)
            force_vals = section_df[force_col].values
            force_N = force_vals * 9.80665
            handle_force = force_N * (outboard_m / oar_length_m)

            # Power = Force * Handle Speed (only when handle speed is positive)
            positive_speed_mask = handle_speed > 0
            instant_power = handle_force[positive_speed_mask] * handle_speed[positive_speed_mask]
            instant_power = instant_power/10
            average_power = np.mean(instant_power) if len(instant_power) > 0 else 0

            watts[f"{force_col}_Watts"] = average_power

            # Compute effective length: handle arc length above threshold
            eff_threshold = 20.0  # N threshold for effective length
            eff_mask = handle_force > eff_threshold
            eff_angles_rad = angles_rad[eff_mask]
            if len(eff_angles_rad) > 0:
                delta_rad = eff_angles_rad.max() - eff_angles_rad.min()
                effective_length = np.rad2deg(inboard_m * delta_rad)
            else:
                effective_length = np.nan

            lengths[f"{angle_col}_EffectiveLength"] = effective_length

        # Append results to lists
        watts_list.append(watts)
        length_list.append(lengths)

    # Convert lists to DataFrames
    watts_df = pd.DataFrame(watts_list)
    length_df = pd.DataFrame(length_list)

    return watts_df, length_df

# Plot normalized data with detected change points
def plot_normalized_data_with_change_points(df, change_points_speed):
    plt.figure(figsize=(14, 7))

    plt.plot(df['Time'].values, df['Speed'].values, label='Normalized Speed', color='blue', linewidth=2)

    for cp in change_points_speed:
        if cp < len(df):
            cp_time = df.iloc[cp]['Time']
            plt.axvline(x=cp_time, color='red', linestyle='--', linewidth=1, alpha=0.7)

    plt.title('Normalized Speed with Detected Change Points', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized Speed', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main function to orchestrate the workflow
def main(input_file, output_folder):
    start_time = time.time()
    df = load_data(input_file)
    print('Data successfully loaded.')

    df_processed = preprocess_data(df, interval_seconds=1)
    print("Data successfully preprocessed.")

    df_normalized = normalize_features(df_processed[['Speed', 'Time']].copy())

    change_points_speed = detect_change_points_single(df_normalized['Speed'], n_bkps=6)
    print(f"Detected change points for Speed: {change_points_speed}")

    save_change_points(df_processed, change_points_speed, 'Speed', output_folder)

    change_points = change_points_speed

    gate_force_cols = [col for col in df_processed.columns if 'GateForceX' in col]
    gate_angle_cols = [col for col in df_processed.columns if 'GateAngle' in col and 'Vel' not in col]
    gate_angle_vel_cols = [col for col in df_processed.columns if 'GateAngleVel' in col]

    gate_force_cols.sort()
    gate_angle_cols.sort()
    gate_angle_vel_cols.sort()

    watts_df, length_df = compute_watts_and_length(df_processed, change_points, gate_force_cols, gate_angle_cols, gate_angle_vel_cols)

    results_df = pd.concat([watts_df.reset_index(drop=True), length_df.reset_index(drop=True)], axis=1)

    # Print the results
    print("Average Watts and Effective Length per Person in Each Section:")
    print(results_df)

    end_time = time.time()
    print(f"Change point detection and computation completed in {end_time - start_time:.2f} seconds.")

    plot_normalized_data_with_change_points(df_normalized, change_points_speed)

# Define input and output paths
input_file = "data/csv_9_10_1.csv"
output_folder = "change_points_output"

# Run the main workflow
if __name__ == "__main__":
    main(input_file, output_folder)
