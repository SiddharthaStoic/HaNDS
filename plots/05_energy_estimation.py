import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your multi-neuron output
df = pd.read_csv('./results/ecg_multi_neuron_output.csv')
seq_id = 0
seq = df[df['sequence_id'] == seq_id].copy()

# Define novelty firing cost
spike_cost = 0.2  # Change if you want to test different firing energy levels

# Energy calculation function
def estimate_energy(response, novelty):
    energy = 0.0
    for t in range(1, len(response)):
        energy += abs(response[t] - response[t - 1])  # internal effort
        if novelty[t] == 1:
            energy += spike_cost  # extra cost for novelty spike
    return energy

# Estimate energy for all neurons
E1 = estimate_energy(seq['response1'].values, seq['novelty1'].values)
E2 = estimate_energy(seq['response2'].values, seq['novelty2'].values)
E3 = estimate_energy(seq['response3'].values, seq['novelty3'].values)

# Plot the results
plt.figure(figsize=(8, 5))
plt.bar(['Neuron A', 'Neuron B', 'Neuron C'], [E1, E2, E3], color=['orange', 'green', 'purple'])
plt.title('Estimated Energy Usage per Neuron')
plt.ylabel('Relative Energy (abstract units)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()