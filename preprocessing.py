import pandas as pd
import numpy as np
import os
import time
import re
import matplotlib.pyplot as plt

# Load the data efficiently and handle multi-level header
def load_data(file_path):
    # Read the CSV file with appropriate encoding, delimiter, and header
    df = pd.read_csv(file_path, encoding='utf-16', delimiter='\t', header=[0, 1], engine='python').iloc[:, :-1]
    
    # Flatten the MultiIndex columns
    df.columns = ['_'.join(col).strip() if col[1] else col[0].strip() for col in df.columns.values]

    # Ensure the first column is 'Time'
    df.columns = ['Time'] + [x for x in df.columns[1:]]

    # Remove leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Convert data to numeric, handling errors by setting invalid entries to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    df = df[(df['Time'] > 2100*1000) & (df['Time'] < 2500*1000)]
    # Drop rows with any NaN values in critical columns
    df = df.dropna(how='any').reset_index(drop=True)
    return df

# Preprocess data: Ensure correct time units and required columns
def preprocess_data(df):
    # Assuming 'Time' is in milliseconds, convert to seconds if necessary
    if df['Time'].max() > 1e6:  # If max time is large, assume it's in milliseconds
        df['Time'] = df['Time'] / 1000.0

    # Identify columns for each rower (assuming 8 rowers)
    gate_force_cols = [col for col in df.columns if re.match(r'^GateForceX_\d+$', col)]
    gate_angle_cols = [col for col in df.columns if re.match(r'^GateAngle_\d+$', col)]
    gate_angle_vel_cols = [col for col in df.columns if re.match(r'^GateAngleVel_\d+$', col)]

    # Ensure we have exactly 8 columns for each
    gate_force_cols = sorted(gate_force_cols)[:8]
    gate_angle_cols = sorted(gate_angle_cols)[:8]
    gate_angle_vel_cols = sorted(gate_angle_vel_cols)[:8]

    # Ensure all required columns are present
    required_cols = ['Time'] + gate_force_cols + gate_angle_cols + gate_angle_vel_cols
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} is missing from the data.")

    # Drop rows with NaN in critical columns
    df = df.dropna(subset=required_cols)

    # Sort by 'Time' in case it's not sorted
    df = df.sort_values('Time').reset_index(drop=True)

    return df, gate_force_cols, gate_angle_cols, gate_angle_vel_cols

# Compute watts and effective length per stroke for each rower
def compute_watts_and_length(df, gate_force_cols, gate_angle_cols, gate_angle_vel_cols):
    # Constants from the logger
    inboard_mm = 1135.0  # mm (Inboard length of the oar)
    oar_length_mm = 3690.0  # mm (Total oar length)
    outboard_mm = oar_length_mm - inboard_mm  # mm (Outboard length of the oar)

    # Convert lengths to meters
    inboard_m = inboard_mm / 1000.0
    outboard_m = outboard_mm / 1000.0
    oar_length_m = oar_length_mm / 1000.0

    # Sampling frequency and time interval
    sample_rate = 50.0  # Hz (Sampling frequency)
    time_interval = 1.0 / sample_rate  # seconds

    # Threshold for detecting strokes
    start_force_threshold = 20.0 # kgf -> N

    # Number of rowers
    num_rowers = len(gate_force_cols)
    
    # Lists to store results
    watts_matrices = [[] for _ in range(num_rowers)]
    length_matrices = [[] for _ in range(num_rowers)]

    # For each rower (gate)
    for idx in range(num_rowers):
        force_col = gate_force_cols[idx]
        angle_col = gate_angle_cols[idx]
        anglev_col = gate_angle_vel_cols[idx]

        # Oarlock force (kgf)
        force_kgf = df[force_col].values

        # Gate angle in degrees
        angles_deg = df[angle_col].values

        # Calculate angular velocity in degrees per second
        angular_velocity_deg_s = df[anglev_col].values

        # Convert angles to radians
        angles_rad = np.deg2rad(angles_deg)

        # Handle Speed Calculation
        # Handle Speed (m/s) = Inboard (m) * Angular Velocity (rad/s) * cos(Angle)
        angular_velocity_rad_s = np.deg2rad(angular_velocity_deg_s)
        handle_speed_m_s = inboard_m * angular_velocity_rad_s * np.cos(angles_rad)

        # Oarlock Force Conversion to N
        oarlock_force_N = force_kgf * 9.80665  # N

        # Handle Force Calculation
        # Handle Force (N) = Oarlock Force (N) * (Outboard (m) / Oar Length (m))
        handle_force_N = oarlock_force_N * (outboard_m / oar_length_m)

        # Stroke Detection
        stroke_starts = np.where(
            (force_kgf[:-1] < start_force_threshold) & (force_kgf[1:] >= start_force_threshold)
        )[0] + 1

        # For the purpose of this calculation, define stroke ends as the next stroke start or end of data
        stroke_ends = np.append(stroke_starts[1:], len(force_kgf) - 1)

        # Ensure the arrays are the same length
        num_strokes = len(stroke_starts)

        # Calculate power and effective length for each stroke
        for i in range(num_strokes):
            start_idx = stroke_starts[i]
            end_idx = stroke_ends[i]

            # Extract stroke data
            stroke_time = (end_idx - start_idx) * time_interval
            stroke_handle_speed = handle_speed_m_s[start_idx:end_idx]
            stroke_handle_force = handle_force_N[start_idx:end_idx]
            stroke_angles = angles_deg[start_idx:end_idx]
            stroke_force = force_kgf[start_idx:end_idx]

            # Only consider when handle speed is positive
            positive_speed_mask = stroke_handle_speed > 0
            positive_speeds = stroke_handle_speed[positive_speed_mask]
            positive_forces = stroke_handle_force[positive_speed_mask]

            # Work Calculation
            instantaneous_power_W = positive_forces * positive_speeds  # W
            total_work_J = np.sum(instantaneous_power_W * time_interval)

            # Average Power Calculation
            if stroke_time > 0:
                power_W = total_work_J / stroke_time
            else:
                power_W = np.nan

            watts_matrices[idx].append(power_W)

            # Effective Length Calculation
            force_threshold = start_force_threshold
            force_above_threshold = stroke_force > force_threshold
            angles_above_threshold = stroke_angles[force_above_threshold]

            if len(angles_above_threshold) > 0:
                effective_length = angles_above_threshold.max() - angles_above_threshold.min()
            else:
                effective_length = np.nan

            length_matrices[idx].append(effective_length)

    # Convert lists to numpy arrays and arrange into matrices
    max_strokes = max(len(watts) for watts in watts_matrices)
    watts_array = np.full((max_strokes, num_rowers), np.nan)
    length_array = np.full((max_strokes, num_rowers), np.nan)

    for idx in range(num_rowers):
        strokes = len(watts_matrices[idx])
        watts_array[:strokes, idx] = watts_matrices[idx]
        length_array[:strokes, idx] = length_matrices[idx]

    return watts_array, length_array

# Plot gate force over time for each rower
def plot_gate_force(df, gate_force_cols):
    plt.figure(figsize=(15, 10))
    for idx, col in enumerate(gate_force_cols):
        plt.plot(df['Time'], df[col], label=f'Rower {idx + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Gate Force X (kgf)')
    plt.title('Gate Force X Over Time for Each Rower')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to orchestrate the workflow
def main(input_file, output_folder):
    start_time = time.time()
    # Load data
    df = load_data(input_file)
    print('Data successfully loaded.')

    # Preprocess data
    df_processed, gate_force_cols, gate_angle_cols, gate_angle_vel_cols = preprocess_data(df)
    print("Data successfully preprocessed.")

    # Plot gate force over time for each rower
    plot_gate_force(df_processed, gate_force_cols)

    # Compute watts and effective length per stroke for each rower
    watts_array, length_array = compute_watts_and_length(df_processed, gate_force_cols, gate_angle_cols, gate_angle_vel_cols)

    # Print the arrays of watts and effective lengths
    print("Watts per stroke array (shape: {}):".format(watts_array.shape))
    print(watts_array)

    print("Effective length per stroke array (shape: {}):".format(length_array.shape))
    print(length_array)

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time:.2f} seconds.")

    # Save the arrays to CSV files
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    watts_df = pd.DataFrame(watts_array, columns=[f'Rower_{i+1}' for i in range(watts_array.shape[1])])
    length_df = pd.DataFrame(length_array, columns=[f'Rower_{i+1}' for i in range(length_array.shape[1])])

    watts_output_file = os.path.join(output_folder, "watts_per_stroke.csv")
    length_output_file = os.path.join(output_folder, "effective_length_per_stroke.csv")

    watts_df.to_csv(watts_output_file, index=False)
    length_df.to_csv(length_output_file, index=False)

    print(f"Watts per stroke saved to {watts_output_file}")
    print(f"Effective length per stroke saved to {length_output_file}")

# Define input and output paths
input_file = "../data_2/csv_10_4_2.csv"  # Replace with your actual data file path
output_folder = "change_points_output"

# Run the main workflow
if __name__ == "__main__":
    main(input_file, output_folder)