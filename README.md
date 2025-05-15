````
  ____ ____  _ _  _  ____   ___     ____                _                   
 / ___/ ___|/ | || ||___ \ / _ \   / ___|__ _ _ __  ___| |_ ___  _ __   ___ 
| |   \___ \| | || |_ __) | | | | | |   / _` | '_ \/ __| __/ _ \| '_ \ / _ \
| |___ ___) | |__   _/ __/| |_| | | |__| (_| | |_) \__ \ || (_) | | | |  __/
 \____|____/|_|  |_||_____|\___/   \____\__,_| .__/|___/\__\___/|_| |_|\___|
                                             |_|                            
````

# Rowing Telemetry Change Point Detection and Power Prediction
This project provides tools for automated analysis of rowing telemetry data, focusing on the detection of effort
segments (pieces) using change point algorithms and the prediction of power output using machine learning models.
Developed as a capstone project, it aims to eliminate the need for manual segmentation of rowing sessions and to provide
accurate, automated insights into rowing performance.

Features
- Automated Change Point Detection: Identifies the start and end of effort segments in rowing sessions using the Pruned
Exact Linear Time (PELT) algorithm.
- Data Preprocessing: Cleans and organizes raw telemetry data, computes derived metrics such as power and effective stroke
length.
- Power Prediction Models: Implements PyTorch-based neural networks to predict average power output, with and without
explicit stroke segmentation features.
- Visualization: Plots normalized speed and detected change points, enabling visual inspection of segmentation accuracy.