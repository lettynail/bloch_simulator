"""
Public API for the bloch_simulator package.

This module re-exports the most useful functions from:
- bloch_simulator_1_qbit.py
- bloch_simulator_2_qbits.py

So examples (and users) can simply:
    from bloch_simulator import continuous_path_1q, animate_trajectory_2q, ...
"""

# ----- 1-qubit exports -----
from .bloch_simulator_1_qbit import (
    # core simulation
    continuous_path,
    animate_trajectory,
    radius,
    fidelity,

    # gates/helpers often used in examples
    RXop, RYop, RZop,
    rm_global_phase,
    unitary_to_axis_angle,
    axis_angle_to_unitary,

    # noise (discretized per step)
    total_noise_steps,
    discreet_depolarizing,
    discreet_T1,
    discreet_T2,

    # common fixed gates (Operator objects)
    X, Y, Z, H, S, T, I,

    # basic states/constructors (optional in examples)
    bloch_vector_rho,
)

# ----- 2-qubit exports -----
from .bloch_simulator_2_qbits import (
    # core simulation
    continuous_path_2q,
    animate_trajectory as animate_trajectory_2q,

    # metrics
    purity,
    entropy_vN,
    concurrence_2q,
    radius as radius_2q,                # 2q radius helper (on Bloch trajectories)
    plot_purity,
    plot_entropy_lists,
    plot_concurrence,

    # building blocks / utilities
    embed2,
    unitary_step,
    bloch_vector_rho_1q,
    bloch_to_rho1,
    reduced_states_from_2q,

    # noise (discretized)
    total_noise_steps,
    discreet_depolarizing as discreet_depolarizing_2q,  # keep naming from source
    discreet_T1 as discreet_T1_2q,
    discreet_T2 as discreet_T2_2q,

    # common fixed gates for 2q module
    X1, Y1, Z1, H1, S1, T1, I1, CX, SWAP, RX1, RY1, RZ1,
)

__all__ = [
    # 1q
    "continuous_path_1q", "animate_trajectory_1q", "radius_1q", "fidelity",
    "RXop", "RYop", "RZop", "rm_global_phase", "unitary_to_axis_angle",
    "axis_angle_to_unitary", "total_noise_steps", "discreet_depolarizing_1q",
    "discreet_T1_1q", "discreet_T2_1q", "X", "Y", "Z", "H", "S", "T", "I",
    "bloch_vector_rho",

    # 2q
    "continuous_path_2q", "animate_trajectory_2q", "purity", "entropy_vN",
    "concurrence_2q", "radius", "plot_purity", "plot_entropy_lists",
    "plot_concurrence", "embed2", "unitary_step", "bloch_vector_rho_1q",
    "bloch_to_rho1", "reduced_states_from_2q", "total_noise_steps_2q",
    "discreet_depolarizing_2q", "discreet_T1_2q", "discreet_T2_2q",
    "X1", "Y1", "Z1", "H1", "S1", "T1", "I1", "CX", "SWAP", "RX1", "RY1", "RZ1",
]
