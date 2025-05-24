# Print entropy scores for each neuron's novelty output
import pandas as pd
import numpy as np
from scipy.stats import entropy

df = pd.read_csv('./results/ecg_multi_neuron_output.csv')
seq_id = 0
seq = df[df['sequence_id'] == seq_id]

def binary_entropy(sequence):
    counts = np.bincount(sequence)
    probs = counts / len(sequence)
    return entropy(probs, base=2)

print(f"Entropy Scores for Sequence {seq_id}:")
print("Neuron A:", round(binary_entropy(seq['novelty1'].tolist()), 4), "bits")
print("Neuron B:", round(binary_entropy(seq['novelty2'].tolist()), 4), "bits")
print("Neuron C:", round(binary_entropy(seq['novelty3'].tolist()), 4), "bits")