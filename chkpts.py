import pandas as pd
import numpy as np
import ruptures as rpt
import os
import time

# Constants for the oar geometry (provide actual values)
INBOARD = 1.135
OUTBOARD = 3.75
OAR_LENGTH = INBOARD + OUTBOARD  # Total oar length
SAMPLING_FREQ = 50  # Hz (data is at 50Hz)
DELTA_TIME = 1 / SAMPLING_FREQ  # Time between samples (0.02 seconds)
GRAVITY = 9.80665  # m/s^2

# Threshold for detecting drive start (force threshold)
FORCE_THRESHOLD = 30

# Load the data efficiently
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-16', delimiter='\t')
    df = df.iloc[1:, :-1].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='any').reset_index(drop=True)
    return df

def preprocess_data(df):
    # Ensure 'Time' is in seconds
    if df['Time'].max() > 1e6:
        df['Time'] = df['Time'] / 1000.0

    # Identify columns
    gate_force_cols = [col for col in df.columns if 'GateForceX' in col]
    gate_angle_cols = [col for col in df.columns if 'GateAngle' in col and 'Vel' not in col]

    required_cols = ['Time', 'Speed'] + gate_force_cols + gate_angle_cols
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} is missing from the data.")

    df = df.dropna(subset=required_cols)

    df = df.sort_values('Time').reset_index(drop=True)

    return df, gate_force_cols, gate_angle_cols

def detect_change_points(df, n_bkpts=6):
    # Use Speed for change point detection
    data_series = df['Speed'].values  # Use 'Speed' as the variable for detection
    model = "l2"  # Model for change point detection
    algo = rpt.Binseg(model=model).fit(data_series)
    change_points = algo.predict(n_bkps=n_bkpts)
    # Remove the last point (end of data)
    if change_points and change_points[-1] == len(data_series):
        change_points = change_points[:-1]
    return change_points

# Detect drive starts using force threshold logic within each piece
def detect_drive_starts(piece_df, gate_force_cols):
    drive_starts = {col: [] for col in gate_force_cols}

    for col in gate_force_cols:
        force = piece_df[col].values
        # Detect where force crosses above the threshold
        above_threshold = force >= FORCE_THRESHOLD
        drive_start_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1

        # Adjust indices to match the piece's indices
        drive_start_indices += piece_df.index[0]

        drive_starts[col] = drive_start_indices

    return drive_starts

# Compute power and effective length per rower per piece
def compute_power_and_effective_length_per_piece(df, gate_force_cols, gate_angle_cols, change_points):
    results_list = []

    for i in range(0, len(change_points) - 1, 2):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        piece_num = i // 2 + 1
        piece_df = df.iloc[start_idx:end_idx]

        # Detect drive starts within the piece
        drive_starts = detect_drive_starts(piece_df, gate_force_cols)

        for idx in range(len(gate_force_cols)):
            force_col = gate_force_cols[idx]
            angle_col = gate_angle_cols[idx]
            starts = drive_starts[force_col]

            # Skip if not enough drive starts in the piece
            if len(starts) < 2:
                continue

            total_work = 0
            total_duration = 0
            total_effective_length = 0
            stroke_count = 0

            for j in range(len(starts) - 1):
                start_idx = starts[j]
                end_idx = starts[j + 1]  # Next drive start

                # Ensure indices are within the piece
                if end_idx > piece_df.index[-1]:
                    break

                # Extract stroke data
                stroke_df = df.iloc[start_idx:end_idx]

                # Extract variables
                times = stroke_df['Time'].values
                angles_deg = stroke_df[angle_col].values
                # Convert raw force (kgf) to Newtons
                forces = stroke_df[force_col].values * GRAVITY

                if len(angles_deg) < 2:
                    continue

                angles_rad = np.deg2rad(angles_deg)

                angular_velocity = np.diff(angles_rad) / DELTA_TIME

                forces = forces[1:]
                angles_rad = angles_rad[1:]
                times = times[1:]

                handle_speed = INBOARD * angular_velocity * np.cos(angles_rad)

                force_on_handle = forces * (OUTBOARD / OAR_LENGTH)

                positive_indices = handle_speed > 0
                if not np.any(positive_indices):
                    continue

                handle_speed_pos = handle_speed[positive_indices]
                force_on_handle_pos = force_on_handle[positive_indices]
                times_pos = times[positive_indices]

                power = handle_speed_pos * force_on_handle_pos

                # Work done during the stroke (Joules)
                work = np.sum(power * DELTA_TIME)

                # Stroke duration (seconds)
                stroke_duration = times[-1] - times[0]

                # Effective Length Calculation: handle arc length for force > threshold
                eff_threshold = 30.0  # N threshold for effective length
                eff_mask = force_on_handle > eff_threshold
                eff_angles_rad = angles_rad[eff_mask]
                if len(eff_angles_rad) > 0:
                    effective_length = eff_angles_rad.max() - eff_angles_rad.min()
                    effective_length = np.rad2deg(effective_length)
                else:
                    effective_length = np.nan

                total_work += work
                total_duration += stroke_duration
                total_effective_length += effective_length
                stroke_count += 1

            if stroke_count > 0 and total_duration > 0:
                average_power = total_work / total_duration
                average_effective_length = total_effective_length / stroke_count
            else:
                average_power = 0
                average_effective_length = 0

            results_list.append({
                'Rower': f'Rower_{idx+1}',
                'Piece': piece_num,
                'AveragePower_Watts': average_power,
                'AverageEffectiveLength_Degrees': average_effective_length
            })

    results_df = pd.DataFrame(results_list)

    return results_df

def main(input_file, output_folder, n_bkpts=6):
    start_time = time.time()

    df = load_data(input_file)
    print('Data successfully loaded.')

    df, gate_force_cols, gate_angle_cols = preprocess_data(df)
    print("Data successfully preprocessed.")

    change_points = detect_change_points(df, n_bkpts=n_bkpts)
    print(f"Detected change points at indices: {change_points}")

    results_df = compute_power_and_effective_length_per_piece(df, gate_force_cols, gate_angle_cols, change_points)
    print("Power and effective length computed for each rower per piece.")

    print("Average Power and Effective Length per Rower per Piece:")
    print(results_df)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    results_df.to_csv(os.path.join(output_folder, "power_and_effective_length_per_piece.csv"), index=False)
    print(f"Results saved to {output_folder}")

    end_time = time.time()
    print(f"Computation completed in {end_time - start_time:.2f} seconds.")

input_file = "data/csv_9_10_1.csv"
output_folder = "output"

# Run the main workflow
if __name__ == "__main__":
    main(input_file, output_folder, n_bkpts=6)
