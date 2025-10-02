#Exportation of the 1 qubit example plots.

import os
import matplotlib   
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from qiskit.quantum_info import Statevector, DensityMatrix
from bloch_simulator.bloch_simulator_1_qbit import (
    H, Y, T, X, S,
    continuous_path, radius, fidelity,
    total_noise_steps, discreet_depolarizing, discreet_T1, discreet_T2,
    animate_trajectory
)

matplotlib.use("Agg")

# --- initial state ---
ket0 = Statevector([1, 0])
rho0 = DensityMatrix(ket0)

# --- gate sequence ---
seq = [H, Y, T, X, H, S]
steps_per_gate = 300
dt = 0.001

# --- noise window & discretized channels ---
n_seq = len(seq)
t_noise_steps = total_noise_steps(0, n_seq, steps_per_gate=steps_per_gate, n_ops=n_seq)

kraus_dep = discreet_depolarizing(0.5, total_steps=t_noise_steps)
kraus_T1  = discreet_T1(t_total=t_noise_steps*dt, T1=0.4, total_steps=t_noise_steps)
kraus_T2  = discreet_T2(t_total=t_noise_steps*dt, T2=0.2, total_steps=t_noise_steps)
kraus_channels = [kraus_dep, kraus_T1, kraus_T2]

# --- trajectories ---
pts0 = continuous_path(rho0, seq, steps_per_gate=steps_per_gate)
pts  = continuous_path(rho0, seq, steps_per_gate=steps_per_gate,
                       kraus_channels=kraus_channels, noise_start=0, noise_end=n_seq)

# --- diagnostics ---

os.makedirs("exports/plots/1_qubit", exist_ok=True)
os.makedirs("exports/animations/1_qubit", exist_ok=True)

radius(pts0, dt, "Radius without noise")
plt.gcf().savefig("exports/plots/1_qubit/1q_radius_noiseless.png", dpi=300, bbox_inches="tight")
plt.close()

radius(pts, dt, "Radius with noise")
plt.gcf().savefig("exports/plots/1_qubit/1q_radius_noisy.png", dpi=300, bbox_inches="tight")
plt.close()

fidelity(pts0, pts, dt, "Fidelity between noisy and noiseless trajectories")
plt.gcf().savefig("exports/plots/1_qubit/1q_fidelity.png", dpi=300, bbox_inches="tight")
plt.close()

# --- animation ---

writer = FFMpegWriter(fps=15, bitrate=1800)

anim0 = animate_trajectory(pts0, interval_ms=10)
anim0.save("exports/animations/1_qubit/1q_noiseless.mp4", writer=writer, dpi = 80)
plt.close()

anim  = animate_trajectory(pts,  interval_ms=10)
anim.save("exports/animations/1_qubit/1q_noisy.mp4", writer=writer, dpi = 80)
plt.close()



