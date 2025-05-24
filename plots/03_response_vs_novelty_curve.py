import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./results/ecg_multi_neuron_output.csv')
seq_id = 0
seq = df[df['sequence_id'] == seq_id].copy()

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

neurons = {
    'Neuron A': ('response1', 'novelty1', 'orange'),
    'Neuron B': ('response2', 'novelty2', 'green'),
    'Neuron C': ('response3', 'novelty3', 'purple')
}

for ax, (title, (resp_col, nov_col, color)) in zip(axes, neurons.items()):
    ax.plot(seq['timestep'], seq[resp_col], label=f'{title} Response', color=color)
    ax.scatter(seq[seq[nov_col] == 1]['timestep'], seq[seq[nov_col] == 1][resp_col],
               color='red', s=30, label='Novelty Triggered')
    ax.set_ylabel("Response")
    ax.legend()
    ax.set_title(f"{title}: Response Curve and Novelty Events")
    ax.grid(True)

axes[-1].set_xlabel("Timestep")
plt.tight_layout()
plt.show()