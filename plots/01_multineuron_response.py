# Plot all three neuron responses with per-neuron novelty markers
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./results/ecg_multi_neuron_output.csv')
seq_id = 0
sequence = df[df['sequence_id'] == seq_id]

plt.figure(figsize=(12, 6))
plt.plot(sequence['timestep'], sequence['value'], label='ECG Signal', color='blue')
plt.plot(sequence['timestep'], sequence['response1'], label='Neuron A', color='orange')
plt.plot(sequence['timestep'], sequence['response2'], label='Neuron B', color='green')
plt.plot(sequence['timestep'], sequence['response3'], label='Neuron C', color='purple')

# Novelty markers
plt.scatter(sequence[sequence['novelty1'] == 1]['timestep'],
            sequence[sequence['novelty1'] == 1]['value'], color='orange', s=20, marker='o', label='Novelty A')
plt.scatter(sequence[sequence['novelty2'] == 1]['timestep'],
            sequence[sequence['novelty2'] == 1]['value'], color='green', s=20, marker='x', label='Novelty B')
plt.scatter(sequence[sequence['novelty3'] == 1]['timestep'],
            sequence[sequence['novelty3'] == 1]['value'], color='purple', s=20, marker='^', label='Novelty C')

plt.title('HaNDS: Multi-Neuron Response with Novelty Detection')
plt.xlabel('Timestep')
plt.ylabel('Signal / Response')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()