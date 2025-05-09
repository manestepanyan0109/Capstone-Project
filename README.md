
 #  Capstone Project: Alcohol Consumption Detection Using Wearable Data

This repository contains the code, data, and results for my Bachelor's Capstone Project at the American University of Armenia. The project explores the use of time-series inertial sensor data to detect alcohol consumption events using machine learning and deep learning techniques.

---

## 📚 Project Overview

**Title:** *Comparative Analysis of Neural Architectures for Alcohol Consumption Detection Using Time-Series Inertial Data*

The goal of this project is to develop models that predict alcohol consumption based on Transdermal Alcohol Concentration (TAC) and accelerometer data. It compares several architectures—FNN, RNN, LSTM, CNN, and Echo State Networks (ESNs)—to evaluate their performance on classification and regression tasks related to sobriety detection.

---

## 📁 Project Structure

### **📊 Data**
- `all_accelerometer_data_pids_13.csv` → Raw accelerometer readings from wearable devices.
- `merged_data.csv` → Merged and cleaned dataset used for training and evaluation.
- `phone_types.csv` → Metadata about device types used in the study.
- `clean_tac.zip` → Cleaned Transdermal Alcohol Concentration (TAC) readings.
- `raw_tac.zip` → Unprocessed/raw TAC sensor data.

### **🧪 Notebooks**
- `Data_preparation.ipynb` → Initial preprocessing, merging sensor and TAC data.
- `Data_processing.ipynb` → Complexity and entropy feature extraction.
- `Reservoir.ipynb` → Echo State Network (ESN) implementation and training.
- `Visual.ipynb` → Visualizations and plots of model predictions.
- `sandbox.ipynb` → Sandbox for quick model testing and debugging.

### **🧠 Experiments**
- (Optionally structured folders) `FNN_model/`, `LSTM_model/`, `CNN_model/`, `RNN_model/`, `ESN_model/` → Model-specific training results and saved models.
- `grid_search_results/` → Logs and outputs from hyperparameter tuning experiments.

### **⚙️ Code**
- `utils.py` → Utility functions for data loading, preprocessing, and metrics.
- `config.yaml` (if used) → Centralized configuration for model parameters and experiment settings.

### **📈 Results**
- `figures/` → Saved plots: training curves, prediction overlays, classification accuracy.
- `metrics/` → CSV files summarizing RMSE, MAE, R², Pearson Correlation, and classification metrics.

### **📜 Other Files**
- `requirements.txt` → List of Python packages required to run the project.
- `.gitignore` → Git exclusion rules.
- `README.md` → This documentation file.

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/manestepanyan0109/Capstone-Project.git
cd Capstone-Project
