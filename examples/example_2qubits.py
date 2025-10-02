# examples/example_2qubits.py
# Minimal usage demo for the 2-qubit simulator.

import matplotlib
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, DensityMatrix
from bloch_simulator import (
    H1, I1, CX,
    embed2, total_noise_steps,
    discreet_depolarizing, discreet_T1, discreet_T2,
    continuous_path_2q, entropy_vN, concurrence_2q,
    radius, plot_purity, plot_entropy_lists, plot_concurrence,
    animate_trajectory
)

matplotlib.use("Qt5Agg")

# --- basis states (2 qubits) ---
ket00 = Statevector([1, 0, 0, 0])
rho00 = DensityMatrix(ket00)

# --- sequence: (I ⊗ H) -> CX -> (I ⊗ H)
seq_A = [H1]
seq_B = [I1]
seq   = embed2(seq_B, seq_A)
seq.append(CX)
seq += embed2(seq_B, seq_A)

steps_per_gate = 300
dt = 0.001
n_seq = len(seq)

# --- noise window ---
t_noise_steps = total_noise_steps(0, n_seq, steps_per_gate=steps_per_gate, n_ops=n_seq)
t_noise = t_noise_steps * dt

# --- discretized channels (apply on qubit A only, identity on B) ---
kraus_T1_A = discreet_T1(t_noise, 0.4, total_steps=t_noise_steps)
kraus_I_B  = [[I1, I1]]  # dummy identity list to match shapes
kraus_channels = [  # lift to 4x4
    [k for k in ( __import__('numpy').kron(a, b) for a, b in zip(kraus_T1_A, kraus_I_B[0]) )]
]

# --- simulate ---
pts0_A, pts0_B, purity0, rhos0 = continuous_path_2q(rho00, seq, steps_per_gate=steps_per_gate, return_rhos=True)
pts_A,  pts_B,  purity,  rhos  = continuous_path_2q(rho00, seq, steps_per_gate=steps_per_gate,
                                                    kraus_channels=kraus_channels,
                                                    noise_start=0, noise_end=n_seq,
                                                    return_rhos=True)

# --- metrics ---
S0_A, S0_B = entropy_vN(pts0_A), entropy_vN(pts0_B)
S_A,  S_B  = entropy_vN(pts_A),  entropy_vN(pts_B)
C0 = [concurrence_2q(r) for r in rhos0]
C  = [concurrence_2q(r) for r in rhos]

# --- plots ---
radius(pts0_A, dt, title="A radius without noise")
radius(pts0_B, dt, title="B radius without noise")
radius(pts_A,  dt, title="A radius with noise")
radius(pts_B,  dt, title="B radius with noise")
plot_purity(purity0, dt, title="System purity without noise")
plot_purity(purity,  dt, title="System purity with noise")
plot_entropy_lists(S0_A, dt, title="S(ρ_A) without noise")
plot_entropy_lists(S0_B, dt, title="S(ρ_B) without noise")
plot_entropy_lists(S_A,  dt, title="S(ρ_A) with noise")
plot_entropy_lists(S_B,  dt, title="S(ρ_B) with noise")
plot_concurrence(C0, dt, title="Concurrence C(t) without noise")
plot_concurrence(C,  dt, title="Concurrence C(t) with noise")

# --- animations ---
anim0 = animate_trajectory(pts0_A, pts0_B, interval_ms=20)
anim  = animate_trajectory(pts_A,  pts_B,  interval_ms=20)
plt.show()

