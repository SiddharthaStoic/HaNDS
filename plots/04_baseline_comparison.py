# Compare HaNDS with classical and ML-based detectors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

df = pd.read_csv('./results/ecg_multi_neuron_output.csv')
seq_id = 0
seq = df[df['sequence_id'] == seq_id].copy()

window = 20
seq['mean'] = seq['value'].rolling(window=window, center=True).mean()
seq['std'] = seq['value'].rolling(window=window, center=True).std()
seq['z_anomaly'] = ((seq['value'] - seq['mean']).abs() / seq['std']) > 2

seq['median'] = seq['value'].rolling(window=window, center=True).median()
seq['mad'] = (seq['value'] - seq['median']).abs().rolling(window=window, center=True).median()
seq['mad_anomaly'] = (seq['value'] - seq['median']).abs() > 3 * seq['mad']

seq['slope'] = seq['value'].diff()
seq['slope_mean'] = seq['slope'].rolling(window=window, center=True).mean()
seq['slope_std'] = seq['slope'].rolling(window=window, center=True).std()
seq['slope_anomaly'] = (seq['slope'] - seq['slope_mean']).abs() > 2 * seq['slope_std']

iso_model = IsolationForest(contamination=0.05, random_state=42)
seq['iso_score'] = iso_model.fit_predict(seq[['value']])
seq['iso_anomaly'] = seq['iso_score'] == -1

svm_model = OneClassSVM(nu=0.05, kernel="rbf", gamma='scale')
seq['svm_score'] = svm_model.fit_predict(seq[['value']])
seq['svm_anomaly'] = seq['svm_score'] == -1

plt.figure(figsize=(14, 8))
plt.plot(seq['timestep'], seq['value'], label='ECG Signal', color='blue')
plt.scatter(seq[seq['novelty1'] == 1]['timestep'], seq[seq['novelty1'] == 1]['value'], color='orange', marker='o', s=30, label='HaNDS A')
plt.scatter(seq[seq['z_anomaly']]['timestep'], seq[seq['z_anomaly']]['value'], color='red', marker='x', s=40, label='Z-Score')
plt.scatter(seq[seq['mad_anomaly']]['timestep'], seq[seq['mad_anomaly']]['value'], color='green', marker='^', s=40, label='MAD')
plt.scatter(seq[seq['slope_anomaly']]['timestep'], seq[seq['slope_anomaly']]['value'], color='purple', marker='s', s=40, label='Slope')
plt.scatter(seq[seq['iso_anomaly']]['timestep'], seq[seq['iso_anomaly']]['value'], color='brown', marker='P', s=40, label='IForest')
plt.scatter(seq[seq['svm_anomaly']]['timestep'], seq[seq['svm_anomaly']]['value'], color='black', marker='D', s=40, label='One-Class SVM')
plt.title(f'HaNDS vs Classical and ML Detectors (Seq {seq_id})')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()