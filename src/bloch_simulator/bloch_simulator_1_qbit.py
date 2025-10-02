####### Imports #######

import os
import numpy as np
from typing import Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from qiskit.circuit.library import XGate, YGate, ZGate, HGate, SGate, TGate, RXGate, RYGate, RZGate
from qiskit.quantum_info import Statevector, Operator, DensityMatrix


####### Circuit Initialisation #######

# Statevectors
ket0 = Statevector([1, 0])
ket1 = Statevector([0, 1])

d0 = DensityMatrix(ket0)
d1 = DensityMatrix(ket1)

def bloch_vector_rho(rho: DensityMatrix) -> Tuple[float, float, float]:
    """
    Returns (x, y, z) = (Tr(ρ X), Tr(ρ Y), Tr(ρ Z)).
    """
    M = np.asarray(rho.data, dtype=complex)
    x = np.real(np.trace(M @ X.data))
    y = np.real(np.trace(M @ Y.data))
    z = np.real(np.trace(M @ Z.data))
    return float(x), float(y), float(z)


####### Gates definition #######

# fixed gates
X = Operator(XGate())
Y = Operator(YGate())
Z = Operator(ZGate())
H = Operator(HGate())
S = Operator(SGate())
T = Operator(TGate())
I = Operator(np.eye(2, dtype=complex))

# parametric rotation gates
def RXop(theta: float) -> Operator: return Operator(RXGate(theta))
def RYop(theta: float) -> Operator: return Operator(RYGate(theta))
def RZop(theta: float) -> Operator: return Operator(RZGate(theta))

def rm_global_phase(op: Operator) -> Operator:
    """removes the global phase of the gate : returns U / sqrt(det(U)) as an Operator."""
    U = np.asarray(op.data, dtype=complex)
    det = np.linalg.det(U)
    if np.isclose(det, 0.0):
        return Operator(U)  
    phase = det**0.5
    return Operator(U / phase)

def unitary_to_axis_angle(op: Operator) -> Tuple[np.ndarray, float]:
    """
    U ∈ U(2) -> (n, theta) such as U ≈ exp(-i * theta/2 * n·σ).
    Returns n (np.array shape (3,)) and theta (float).
    """
    U0 = np.asarray(op.data, dtype=complex)
    U = np.asarray(rm_global_phase(Operator(U0)).data, dtype=complex)

    tr = np.trace(U)
    c = np.clip(np.real(tr)/2.0, -1.0, 1.0)
    theta = 2.0 * np.arccos(c)

    s = np.sin(theta/2.0)
    if np.isclose(s, 0.0, atol=1e-12):
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0

    U01, U10 = U[0,1], U[1,0]
    U00, U11 = U[0,0], U[1,1]

    nx = (1.0/s) * ( np.imag(U01 + U10) )
    ny = (1.0/s) * ( np.real(U10 - U01) )
    nz = (1.0/s) * ( np.imag(U00 - U11) )

    n = np.array([nx, ny, nz], dtype=float)
    nr = np.linalg.norm(n)
    n = np.array([1.0, 0.0, 0.0], dtype=float) if nr < 1e-12 else (n / nr)
    return n, float(theta)

def axis_angle_to_unitary(n, theta) -> Operator:
    """
    Builds exp(-i * theta/2 * n·σ) and returns it as an Operator.
    """
    n = np.asarray(n, dtype=float)
    nr = np.linalg.norm(n)
    if nr == 0.0:
        return I

    n = n / nr
    c = np.cos(theta/2.0)
    s = np.sin(theta/2.0)

    # n·σ as matrices then rewrapped as an Operator
    N = n[0]*X.data + n[1]*Y.data + n[2]*Z.data
    U = c*I.data - 1j*s*N
    return Operator(U)


####### Kraus noise channels definition #######

def kraus_depolarizing(p: float) -> List[np.ndarray]:
    """
    Channel:  E(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    Returns [K0, K1, K2, K3]
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must belong to [0,1].")
    I = np.eye(2, dtype=complex)
    X = Operator(XGate()).data
    Y = Operator(YGate()).data
    Z = Operator(ZGate()).data
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p / 3) * X
    K2 = np.sqrt(p / 3) * Y
    K3 = np.sqrt(p / 3) * Z
    return [K0, K1, K2, K3]

def kraus_T1(t: float, T1: float) -> List[np.ndarray]:
    """
    Channel T1 : amplitude damping to |0⟩ with γ = 1 - exp(-t/T1).
    K0 = [[1, 0], [0, sqrt(1-γ)]],  K1 = [[0, sqrt(γ)], [0, 0]]
    Returns [K0, K1]
    """
    if T1 <= 0:
        raise ValueError("T1 must be > 0.")
    if t < 0:
        raise ValueError("t must be ≥ 0.")
    gamma = 1.0 - np.exp(-t / T1)
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return [K0, K1]

def kraus_T2(t: float, T2: float) -> List[np.ndarray]:
    """
    Channel T2 : phase damping with λ = 1 - exp(-t/T2).
    Returns 3-Kraus standard :
      K0 = sqrt(1-λ) I
      K1 = sqrt(λ) |0⟩⟨0|
      K2 = sqrt(λ) |1⟩⟨1|
    """
    if T2 <= 0:
        raise ValueError("T2 must be > 0.")
    if t < 0:
        raise ValueError("t must be ≥ 0.")
    lam = 1.0 - np.exp(-t / T2)

    I = np.eye(2, dtype=complex)
    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

    K0 = np.sqrt(1.0 - lam) * I
    K1 = np.sqrt(lam) * P0
    K2 = np.sqrt(lam) * P1
    return [K0, K1, K2]

####### Gates application #######

def evolve_rho(rho: DensityMatrix, U: Operator) -> DensityMatrix:
    """ρ ↦ U ρ U†"""
    M0 = np.asarray(rho.data, dtype=complex)
    Umat = np.asarray(U.data, dtype=complex)
    M = Umat @ M0 @ Umat.conj().T
    return DensityMatrix(M)

def evolve_seq_rho(rho: DensityMatrix, ops) -> DensityMatrix:
    """ρ ↦ U ρ U†"""
    M = np.asarray(rho.data, dtype=complex)
    for op in ops:
        opmat = np.asarray(op.data, dtype=complex)
        M = opmat @ M @ opmat.conj().T
    return DensityMatrix(M)

def apply_kraus(rho: DensityMatrix, kraus_ops: Iterable[np.ndarray]) -> DensityMatrix:
    """
    ρ ↦ ∑_k K_k ρ K_k†
    kraus_ops: list 2x2 matrices (numpy) or objects with .__array__()
    """
    M = np.asarray(rho.data, dtype=complex)
    acc = np.zeros_like(M, dtype=complex)
    for K in kraus_ops:
        Kmat = np.asarray(K, dtype=complex)
        acc += Kmat @ M @ Kmat.conj().T
    return DensityMatrix(acc)


####### Continuous trajectories #######
# --- Total steps for the noise window ---
def total_noise_steps(noise_start: int, noise_end: int, steps_per_gate: int, n_ops: int) -> int:
    if n_ops <= 0:
        return 0
    noise_start = max(0, int(noise_start))
    noise_end   = min(n_ops - 1, int(noise_end))
    if noise_end < noise_start:
        return 0
    steps_per_gate = max(1, int(steps_per_gate))
    return (noise_end - noise_start + 1) * steps_per_gate

def discreet_depolarizing(p_total: float, total_steps: int) -> List[np.ndarray]:
    """
    E_total(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    Composition -> p_step = 1 - (1 - p_total)^(1/total_steps)
    Returns the list [K0, K1, K2, K3] for total_steps steps.
    """
    if total_steps <= 0 or p_total <= 0.0:
        # no effective noise
        I = np.eye(2, dtype=complex)
        return [I]  # we return the identity
    p_step = 1.0 - (1.0 - float(p_total))**(1.0/float(total_steps))
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    K0 = np.sqrt(1.0 - p_step) * I
    K1 = np.sqrt(p_step/3.0) * X
    K2 = np.sqrt(p_step/3.0) * Y
    K3 = np.sqrt(p_step/3.0) * Z
    return [K0, K1, K2, K3]

def discreet_T1(t_total: float, T1: float, total_steps: int) -> List[np.ndarray]:
    """
    γ_total = 1 - exp(-t_total/T1)
    Composition -> γ_step = 1 - (1 - γ_total)^(1/total_steps)
    Returns [K0, K1] for total_steps steps.
    """
    if total_steps <= 0 or t_total <= 0.0 or T1 <= 0.0:
        # no effective noise
        I = np.eye(2, dtype=complex)
        return [I] # we return the identity
    gamma_total = 1.0 - np.exp(-float(t_total)/float(T1))
    gamma_step  = 1.0 - (1.0 - gamma_total)**(1.0/float(total_steps))
    K0 = np.array([[1.0, 0.0],[0.0, np.sqrt(1.0 - gamma_step)]], dtype=complex)
    K1 = np.array([[0.0, np.sqrt(gamma_step)],[0.0, 0.0]], dtype=complex)
    return [K0, K1]

def discreet_T2(t_total: float, T2: float, total_steps: int) -> List[np.ndarray]:
    """
    λ_total = 1 - exp(-t_total/T2)
    Composition -> λ_step = 1 - (1 - λ_total)^(1/total_steps)
    Returns [K0, K1, K2] for total_steps steps.
    """
    if total_steps <= 0 or t_total <= 0.0 or T2 <= 0.0:
        # no effective noise
        I = np.eye(2, dtype=complex)
        return [I] # we return the identity
    lam_total = 1.0 - np.exp(-float(t_total)/float(T2))
    lam_step  = 1.0 - (1.0 - lam_total)**(1.0/float(total_steps))
    I  = np.eye(2, dtype=complex)
    P0 = np.array([[1.0, 0.0],[0.0, 0.0]], dtype=complex)
    P1 = np.array([[0.0, 0.0],[0.0, 1.0]], dtype=complex)
    K0 = np.sqrt(1.0 - lam_step) * I
    K1 = np.sqrt(lam_step) * P0
    K2 = np.sqrt(lam_step) * P1
    return [K0, K1, K2]

def continuous_path(
    rho0: DensityMatrix,
    ops: Iterable[Operator],
    *,
    steps_per_gate: int = 60,
    kraus_channels: Optional[List[Iterable]] = None,
    noise_start: int = 0,
    noise_end: Optional[int] = None,
) -> List[Tuple[float, float, float]]:
    """
    Generates (x,y,z) points of the 'Bloch ball' for the evolution of the initial state:
      - Continuous unitaries application: each gate U_k is cut into `steps_per_gate` little rotations
        thanks to axis-angle: U_k ≈ [exp(-i dθ/2 n·σ)]^steps_per_gate
      - Continuous noise: each channel is cut into 't_noise' steps and if the number k of the gate is in [noise_start, noise_end],
        we apply all the channels from 'kraus_channels'  at each step
        (each channel is a list of kraus operators), which makes the point enter the sphere progressively
        during the application of the concerned gates.

    Parameters:
      rho0            : initial state (DensityMatrix)
      ops             : list/iterable of Operators (1-qubit gates)
      steps_per_gate  : number of steps per gate (fluidity)
      kraus_channels  : channels list; each channel = iterable of Kraus operators (2x2)
                        (if None ou [], no decoherence applied)
                        NOTE: the given channels must be incremental (cut into small steps);
                        they are applied at each gate step.
      noise_start     : index (0-based) of the first gate concerned by the noise(0 by default)
      noise_end       : index (0-based) of the last gate concerned by the noise
                        (len(ops)-1 by default)

    Returns:
      points : list[tuple[float,float,float]]  — list of (x,y,z)
    """
    # Normalization of the parametres
    ops = list(ops)
    n_gates = len(ops)
    if n_gates == 0:
        return [bloch_vector_rho(rho0)]  # just the initial point

    if noise_end is None:
        noise_end = n_gates - 1
    noise_start = max(0, int(noise_start))
    noise_end   = min(n_gates - 1, int(noise_end))
    apply_noise_window = (kraus_channels is not None) and (len(kraus_channels) > 0) and (noise_start <= noise_end)

    # Prepare
    rho = rho0
    points = [bloch_vector_rho(rho)]
    steps_per_gate = max(1, int(steps_per_gate))

    # Evolution gate per gate
    for gate_idx, U in enumerate(ops):
        # Axis-angle of the current gate
        n, theta = unitary_to_axis_angle(U)
        if np.isclose(theta, 0.0):
            # No effective rotation -> apply only the noise if the noise window is active on that gate
            if apply_noise_window and (noise_start <= gate_idx <= noise_end):
                for _ in range(steps_per_gate):
                    for channel in kraus_channels:
                        rho = apply_kraus(rho, channel)
                    points.append(bloch_vector_rho(rho))
            continue

        # Build the little rotation only once for that gate
        dtheta = theta / steps_per_gate
        Uk = axis_angle_to_unitary(n, dtheta)  # -> Operator

        # Apply the 'steps_per_gate' steps
        for _ in range(steps_per_gate):
            # 1) Unitary step
            rho = evolve_rho(rho, Uk)

            # 2) Noise if the gate index is in the noise window [noise_start, noise_end]
            if apply_noise_window and (noise_start <= gate_idx <= noise_end):
                for channel in kraus_channels:
                    rho = apply_kraus(rho, channel)

            # 3) Sampling the point
            points.append(bloch_vector_rho(rho))

    return points


####### Physical Interpretation curves #######

def radius(points: Iterable[Tuple[float, float, float]], dt: float = 1.0, title="Radius |r|(t)") -> List[float]:
    """
    Calculate the radius |r| for each point (x,y,z) and plots |r|(t).

    Args:
        points: list of points (x,y,z).
        dt: time step between each points.

    Returns:
        radius: list of each points radius.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("`points` must have shape (N, 3).")

    # Radius for each point
    radius = np.sqrt(np.sum(pts**2, axis=1)).tolist()

    # Time axis: one point = one time t
    t = np.arange(len(radius)) * float(dt)

    # Plot
    plt.figure(figsize=(6, 3.2))
    plt.plot(t, radius, linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Radius |r|")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.25)

    return radius

def fidelity(points_r: Iterable[Tuple[float,float,float]],
             points_s: Iterable[Tuple[float,float,float]],
             dt: float = 1.0, title="Fidelity F(t)") -> List[float]:
    """
    Calculate the fidelity F(rho, sigma) between two trajectories (points in the Bloch Ball) using Bures/Uhlmann formula: F=0.5(1+r⋅s+sqrt((1-|r|2)(1-|s|2))).
    Each point is a tuple(x,y,z).
    
    Args:
        points_r: list of points (x,y,z) for the first trajectory.
        points_s: list de points (x,y,z) for the second trajectory.
        dt: time step between each point.

    Returns:
        F: list of the fidelity for each point.
    """
    R = np.asarray(points_r, dtype=float)
    S = np.asarray(points_s, dtype=float)
    if R.shape != S.shape or R.ndim != 2 or R.shape[1] != 3:
        raise ValueError("The two lists must have shape (N, 3).")

    # Norm and dot product
    dot_rs = np.sum(R * S, axis=1)
    nr = np.minimum(np.sum(R**2, axis=1), 1.0)      # clip the norm to 1
    ns = np.minimum(np.sum(S**2, axis=1), 1.0)

    inside = (1-nr) * (1-ns)
    inside = np.clip(inside, 0.0, None)         # negatives → 0
    F = 0.5 * (1 + dot_rs + np.sqrt(inside))
    F = np.clip(F, 0.0, 1.0)  # clip the fidelity between 0 and 1
    F = F.tolist()

    # Time axis
    t = np.arange(len(F)) * float(dt)

    # Plot
    plt.figure(figsize=(6, 3.2))
    plt.plot(t, F, linewidth=2)
    plt.xlabel("Time (step)")
    plt.ylabel("Fidelity")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.25)

    return F


####### Animation #######

def animate_trajectory(points, interval_ms=25, show_trail=True) -> FuncAnimation:
    if not points:
        raise ValueError("List 'points' is empty.")
    pts = np.asarray(points, dtype=float)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection="3d")

    # Sphere
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.12, linewidth=0)

    # Axis
    ax.plot([-1,1],[0,0],[0,0]); ax.text(1.1,0,0,"X")
    ax.plot([0,0],[-1,1],[0,0]); ax.text(0,1.1,0,"Y")
    ax.plot([0,0],[0,0],[-1,1]); ax.text(0,0,1.1,"Z")
    ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # Animated elements
    scat = ax.scatter([pts[0,0]], [pts[0,1]], [pts[0,2]], s=50, c="red")
    line, = ax.plot([], [], [], linewidth=1.5, alpha=0.7)

    def update(i):
        x,y,z = pts[i]
        scat._offsets3d = ([x], [y], [z])
        if show_trail:
            Xs, Ys, Zs = pts[:i+1,0], pts[:i+1,1], pts[:i+1,2]
            line.set_data(Xs, Ys); line.set_3d_properties(Zs)
        ax.set_title(f"Frame {i+1}/{len(pts)}")
        return scat, line

    anim = FuncAnimation(fig, update, frames=len(pts), interval=interval_ms, blit=False, repeat=True)
    return anim
