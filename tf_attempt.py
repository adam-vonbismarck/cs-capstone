import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, Concatenate, TimeDistributed, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

# Constants
NUM_ROWERS = 8
FEATURES_PER_ROWER = 3  # GateForceX, GateAngle, GateAngleVel

# Step 1: Data Preparation
def load_data(data_folder):
    """
    Load the long-form matrix data and corresponding average watt vectors.
    Each sample consists of time-series data for each rower and an 8-length vector of average watts.
    """
    X_data = []  # List to hold input sequences
    y_data = []  # List to hold output vectors

    # Loop over all files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            df = pd.read_csv(file_path, encoding='utf-16', delimiter='\t', header=[0, 1], engine='python').iloc[:, :-1]
            df.columns = ['_'.join(col).strip() if col[1] else col[0].strip() for col in df.columns.values]

            # Ensure the first column is 'Time'
            df.columns = ['Time'] + [x for x in df.columns[1:]]

            # Remove leading/trailing whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Convert data to numeric, handling errors by setting invalid entries to NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Drop rows with any NaN values in critical columns
            df = df.dropna(how='any').reset_index(drop=True)

            # Extract features for each rower
            sample_data = []
            for i in range(1, NUM_ROWERS + 1):
                # Columns for the current rower
                force_col = f'GateForceX_{i}'
                angle_col = f'GateAngle_{i}'
                angle_vel_col = f'GateAngleVel_{i}'

                # Check if columns exist
                if all(col in df.columns for col in [force_col, angle_col, angle_vel_col]):
                    # Extract the time-series data for the rower
                    rower_data = df[[force_col, angle_col, angle_vel_col]].values
                    sample_data.append(rower_data)
                else:
                    # If any column is missing, fill with zeros
                    sample_data.append(np.zeros((len(df), FEATURES_PER_ROWER)))

            # Stack the data for all rowers (time_steps, num_rowers, features)
            sample_data = np.stack(sample_data, axis=1)  # Shape: (time_steps, num_rowers, features)

            # Load the corresponding average watt vector
            # Assuming the average watts are stored in a separate file or computed beforehand
            # For this example, we'll create a dummy watt vector
            if filename == "csv_10_4_1.csv":
                average_watts = [340, 302, 314, 349, 317, 333, 314, 286]
            if filename == "csv_10_4_2.csv":
                average_watts = [377, 333, 309, 374, 330, 353, 335, 322]
            if filename == "csv_10_4_3.csv":
                average_watts = [415, 331, 370, 416, 369, 349, 343, 335]

            # Append to lists
            X_data.append(sample_data)
            y_data.append(average_watts)

    return X_data, y_data

def preprocess_data(X_data, max_sequence_length=None):
    """
    Pad sequences to the same length and concatenate features for all rowers.
    """
    # Determine the maximum sequence length if not provided
    if max_sequence_length is None:
        max_sequence_length = max([x.shape[0] for x in X_data])

    # Pad sequences
    X_padded = []
    for x in X_data:
        # x shape: (time_steps, num_rowers, features)
        padded_x = pad_sequences(
            x,
            maxlen=max_sequence_length,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )
        # Flatten the features of all rowers into a single vector for each time step
        padded_x = padded_x.reshape((padded_x.shape[0], -1))  # Shape: (time_steps, num_rowers * features_per_rower)
        X_padded.append(padded_x)

    X_padded = np.array(X_padded)  # Shape: (num_samples, time_steps, num_rowers * features_per_rower)

    return X_padded

def build_model(time_steps, features_per_rower, num_rowers):
    """
    Build a simplified LSTM model that processes the combined data for all rowers.
    """
    # Input layer for all rowers combined
    input_layer = Input(shape=(time_steps, num_rowers * features_per_rower), name='combined_input')

    # Masking to handle padded sequences
    masked_input = Masking(mask_value=0.0)(input_layer)

    # LSTM layer
    lstm_out = LSTM(128)(masked_input)

    # Dense layer to output an 8-length vector
    output = Dense(num_rowers, activation='linear', name='output_vector')(lstm_out)

    # Build the model
    model = Model(inputs=input_layer, outputs=output)
    return model

# Step 4: Training
def train_model(model, X_train, y_train, epochs=50, batch_size=16):
    """
    Compile and train the model.
    """
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def main():
    data_folder = 'data'  # Replace with your data folder path

    # Load data
    X_data, y_data = load_data(data_folder)
    y_data = np.array(y_data)  # Shape: (num_samples, num_rowers)

    # Preprocess data
    X_padded = preprocess_data(X_data)
    num_samples, time_steps, features = X_padded.shape

    # Build the model
    model = build_model(time_steps, features // NUM_ROWERS, NUM_ROWERS)
    model.summary()

    # Train the model
    model = train_model(model, X_padded, y_data)

    # Save the model
    model.save('rower_power_prediction_model.h5')
    print("Model saved successfully.")

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
