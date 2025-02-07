# AI Network Traffic Monitoring and Anomaly Detection Dashboard

This project implements an **Advanced Network Anomaly Detection System** using **multiple machine learning models** for detecting suspicious network activities. It is a real-time network traffic monitoring and anomaly detection system built using Dash. It captures network statistics, processes the data, detects anomalies, and visualizes network activity through an interactive web-based dashboard.

## Overview  
The system leverages an **ensemble approach** that combines:  
- **Isolation Forest** (Unsupervised Anomaly Detection)  
- **Random Forest Classifier** (Supervised Learning)  
- **LSTM Neural Network** (Time-series based anomaly detection)  
- **Autoencoder** (Reconstruction-based anomaly detection)  

## Features 
- **Data Preprocessing** – Feature engineering and scaling  
- **Multi-Model Training** – Trains multiple models for robust detection  
- **Ensemble Prediction** – Combines model outputs for accurate anomaly detection  
- **Threshold-Based Anomaly Detection** – Uses MSE for Autoencoder predictions
- **Real-time network traffic monitoring**: Captures packets and network statistics using `psutil`.
- **Anomaly detection**: Identifies abnormal network behavior based on traffic volume.
- **Live dashboard**: Visualizes network traffic and alerts using Dash and Plotly.
- **CSV Logging**: Stores network traffic and connection details for analysis.

## Technologies Used
- Python
- Dash (Plotly)
- Pandas
- Tensorflow
- Scikit-Learn
- Psutil
- Logging
- CSV Handling

## Installation  
Clone the repository:  
```bash
git clone https://github.com/ayomidee-e/AI-ML-Cybersecurity-Projects/tree/master/AI-NetworkAnomaly-Detection.git
cd network-monitor
````

Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

Install dependencies:
````
pip install -r requirements.txt
````

## Usage

1. Run the real-time analysis engine:
   ```sh
   python main.py
   ```
2. Start the Dash dashboard:
   ```sh
   python app.py
   ```
3. Open a web browser and navigate to:
   ```
   http://127.0.0.1:8050/
   ```

## Configuration
- Modify `main.py` to adjust anomaly detection thresholds.
- Update `app.py` to customize dashboard visualizations.

## Contributing
Pull requests are welcome! If you have suggestions, feel free to open an issue.




