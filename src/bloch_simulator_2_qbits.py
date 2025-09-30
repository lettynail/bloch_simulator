####### Imports #######
import numpy as np
from typing import Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from qiskit.quantum_info import Operator, DensityMatrix, partial_trace, Statevector
from qiskit.circuit.library import (
    XGate, YGate, ZGate, HGate, SGate, TGate, RXGate, RYGate, RZGate, CXGate, SwapGate
)
from scipy.linalg import logm, expm


####### Operators #######

# 1-qubit Operators (2x2)
X1 = Operator(XGate())
Y1 = Operator(YGate())
Z1 = Operator(ZGate())
H1 = Operator(HGate())
S1 = Operator(SGate())
T1 = Operator(TGate())
I1 = Operator(np.eye(2, dtype=complex))
def RX1(theta: float) -> Operator: return Operator(RXGate(theta))
def RY1(theta: float) -> Operator: return Operator(RYGate(theta))
def RZ1(theta: float) -> Operator: return Operator(RZGate(theta))

# 2-qubit non-local Operators (4x4)
CX = Operator(CXGate())
SWAP = Operator(SwapGate())

# Embedding 2 1-qubit gates (Ua acts on the first qubit, Ub on the second)
def embed2(OpsA: Iterable[Operator], OpsB: Iterable[Operator]) -> Iterable[Operator]:
    """Returns Ua ⊗ Ub."""
    OpsA = list(OpsA); OpsB = list(OpsB)
    if len(OpsA) != len(OpsB):
        raise ValueError ("OpsA and OpsB must have same length")
    return  [Operator(np.kron(a.data, b.data)) for a, b in zip(OpsA, OpsB)]

# Remove the global phase of an operator
def rm_global_phase(U: np.ndarray) -> np.ndarray:
    # Removes the global phase of U ∈ U(n): U' = U / det(U)^(1/n)
    detU = np.linalg.det(U)
    if np.isclose(detU, 0.0):
        return U
    n = U.shape[0]
    return U / (detU**(1.0/n))

#factorizing (if its possible) a matrix 4x4 into 2 matrices 2x2
def factorize_kron_2x2(U4: np.ndarray, tol=1e-10) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    # Cut U4 into 4 blocks 2x2
    B00 = U4[0:2, 0:2]; B01 = U4[0:2, 2:4]
    B10 = U4[2:4, 0:2]; B11 = U4[2:4, 2:4]
    # if U4 = A⊗B, then each Bxy is proportional to B00
    # take the first non zero block as a reference
    ref = None
    for B in (B00, B01, B10, B11):
        if np.linalg.norm(B) > tol:
            ref = B; break
    if ref is None:
        return None, None  # Null block, not likely for a unitary gate
    # scale coefficients
    coeffs = []
    for B in (B00, B01, B10, B11):
        if np.linalg.norm(B) <= tol:
            coeffs.append(0.0)
        else:
            # scalar c such as B ≈ c*ref 
            c = np.vdot(ref, B) / np.vdot(ref, ref)
            if np.linalg.norm(B - c*ref) > tol*np.linalg.norm(B):
                return None, None
            coeffs.append(c)
    c00, c01, c10, c11 = coeffs
    # Rebuild A = [[c00, c01],[c10, c11]] and B = ref (up to a global phase)
    A = np.array([[c00, c01],[c10, c11]], dtype=complex)
    B = ref
    # Globally normalize U4 ≈ A⊗B -> remove the global phase
    phase = np.linalg.det(A)**0.5 * np.linalg.det(B)**0.5
    if np.abs(phase) > tol:
        A = A / (np.linalg.det(A)**0.5)
        B = B / (np.linalg.det(B)**0.5)
    return A, B



####### Bloch Coordinates #######

def bloch_vector_rho_1q(rho1: DensityMatrix) -> Tuple[float,float,float]:
    X = X1.data; Y = Y1.data; Z = Z1.data
    M = np.asarray(rho1.data, dtype=complex)
    x = float(np.real(np.trace(M @ X)))
    y = float(np.real(np.trace(M @ Y)))
    z = float(np.real(np.trace(M @ Z)))
    return (x, y, z)

def bloch_to_rho1(rho_list: Iterable[Tuple[float,float,float]]) -> np.ndarray:
    """Rebuild ρ = (I + xX + yY + zZ)/2 from its Bloch coordinates (x,y,z)."""
    Rhos = []
    for xyz in rho_list:
        x, y, z = xyz
        I = np.eye(2, dtype=complex)
        X = np.array([[0,1],[1,0]], dtype=complex)
        Y = np.array([[0,-1j],[1j,0]], dtype=complex)
        Z = np.array([[1,0],[0,-1]], dtype=complex)
        Rhos.append(0.5*(I + x*X + y*Y + z*Z))
    return Rhos

def reduced_states_from_2q(rho2: DensityMatrix) -> Tuple[DensityMatrix, DensityMatrix]:
    # Partial trace: we keep A (qubit 0) -> trace out B (qubit 1); and vice versa
    rhoA = partial_trace(rho2, [1])  # trace out qubit B
    rhoB = partial_trace(rho2, [0])  # trace out qubit A
    return DensityMatrix(rhoA), DensityMatrix(rhoB)


####### Discreet unitaries #######



def unitary_step(U: Operator, steps: int) -> Operator:
    stps= max(1, int(steps))
    U4 = rm_global_phase(np.asarray(U.data, complex))
    A, B = factorize_kron_2x2(U4)
    if A is not None and B is not None:
        # locally discretize if we can factorize U into two 2x2 matrices
        A = rm_global_phase(A)
        B = rm_global_phase(B)
        LA, LB = logm(A), logm(B)
        A_step = expm(LA/stps)
        B_step = expm(LB/stps)
        return Operator(np.kron(A_step, B_step))
    # if not, fall back to global discretization with logm/expm
    L = logm(U4)
    return Operator(expm(L/stps))



####### Discreet Kraus noise #######

### for each channel, its main parameter is cut into a smaller parameter such as 'total_noise_steps' composition
### of each channel falls back on the original channel

def total_noise_steps(noise_start: int, noise_end: int, steps_per_gate: int, n_ops: int) -> int:
    if n_ops <= 0: return 0
    noise_start = max(0, int(noise_start))
    noise_end   = min(n_ops - 1, int(noise_end))
    if noise_end < noise_start: return 0
    return (noise_end - noise_start + 1) * max(1, int(steps_per_gate))

def discreet_depolarizing(p_total: float, total_steps: int) -> List[np.ndarray]:
    if total_steps <= 0 or p_total <= 0.0:
        return [np.eye(2, dtype=complex)]
    p_step = 1.0 - (1.0 - float(p_total))**(1.0/float(total_steps))
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    return [np.sqrt(1-p_step)*I, np.sqrt(p_step/3)*X, np.sqrt(p_step/3)*Y, np.sqrt(p_step/3)*Z]

def discreet_T1(t_total: float, T1: float, total_steps: int) -> List[np.ndarray]:
    if total_steps <= 0 or t_total <= 0.0 or T1 <= 0.0:
        return [np.eye(2, dtype=complex)]
    gamma_total = 1.0 - np.exp(-float(t_total)/float(T1))
    gamma_step  = 1.0 - (1.0 - gamma_total)**(1.0/float(total_steps))
    K0 = np.array([[1.0, 0.0],[0.0, np.sqrt(1.0 - gamma_step)]], dtype=complex)
    K1 = np.array([[0.0, np.sqrt(gamma_step)],[0.0, 0.0]], dtype=complex)
    return [K0, K1]

def discreet_T2(t_total: float, T2: float, total_steps: int) -> List[np.ndarray]:
    if total_steps <= 0 or t_total <= 0.0 or T2 <= 0.0:
        return [np.eye(2, dtype=complex)]
    lam_total = 1.0 - np.exp(-float(t_total)/float(T2))
    lam_step  = 1.0 - (1.0 - lam_total)**(1.0/float(total_steps))
    I  = np.eye(2, dtype=complex)
    P0 = np.array([[1.0, 0.0],[0.0, 0.0]], dtype=complex)
    P1 = np.array([[0.0, 0.0],[0.0, 1.0]], dtype=complex)
    return [np.sqrt(1-lam_step)*I, np.sqrt(lam_step)*P0, np.sqrt(lam_step)*P1]


####### Kraus noise application #######

def embed_kraus(kraus_1q_A: Iterable[np.ndarray], kraus_1q_B: Iterable[np.ndarray]) -> List[np.ndarray]:
    if len(kraus_1q_A) != len(kraus_1q_B):
         raise ValueError("kraus_1q_A and kraus_1q_B must have the same length")
    return [np.kron(np.asarray(A,complex), np.asarray(B,complex))for A,B in zip(kraus_1q_A, kraus_1q_B)]


def apply_kraus_2q(rho2: DensityMatrix, kraus_2q: Iterable[np.ndarray]) -> DensityMatrix:
    M = np.asarray(rho2.data, dtype=complex)
    acc = np.zeros_like(M, dtype=complex)
    for K in kraus_2q:
        K = np.asarray(K, dtype=complex)
        acc += K @ M @ K.conj().T
    return DensityMatrix(acc)

####### Physical Interpretation Curves #######

def radius(points: Iterable[Tuple[float, float, float]], dt: float =1.0, title="Radius |r|(t)") -> List[float]:
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
    plt.title("Radius Evolution in the Bloch Ball")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.25)
    plt.show()
    return radius

def purity(rho: DensityMatrix) -> float:
    M = np.asarray(rho.data, dtype=complex)
    return float(np.real(np.trace(M @ M)))

def concurrence_2q(rho2: DensityMatrix | np.ndarray) -> float:
    """
    Wootters concurrence for a 2 qubits state (ρ 4x4).
    Returns C ∈ [0,1].
    """
    M = np.asarray(rho2.data if isinstance(rho2, DensityMatrix) else rho2, dtype=complex)
    # (σ_y ⊗ σ_y)
    sy = np.array([[0, -1j],[1j, 0]], dtype=complex)
    YY = np.kron(sy, sy)
    # R = ρ (Y⊗Y) ρ* (Y⊗Y)  (equivalent formula)
    R = M @ YY @ M.conj() @ YY
    # Non negative eigenvalues (we clip them numerically)
    evals = np.linalg.eigvals(R)
    evals = np.clip(np.real(evals), 0.0, None)
    rts = np.sqrt(np.sort(evals)[::-1])  # λ1 ≥ λ2 ≥ λ3 ≥ λ4
    C = rts[0] - rts[1] - rts[2] - rts[3]
    return float(max(0.0, C))

def entropy_vN(pts_A, base: float = 2.0) -> List[float]:
    """
    Von Neumann entropy S(ρ) = -Tr(ρ log ρ).
    - 'pts_A' list of Bloch coordinates corresponding to a trajectory
    - `base`=2 for bits, =np.e for nats.
    """
    SvN = []
    for rho in bloch_to_rho1(pts_A):
        M = np.asarray(rho.data if isinstance(rho, DensityMatrix) else rho, dtype=complex)
        # Real and positive eigenvalues (numeric robustness)
        evals = np.linalg.eigvalsh((M + M.conj().T)/2)
        evals = np.clip(np.real(evals), 0.0, 1.0)
        nz = evals[evals > 0]
        log_fn = np.log2 if base == 2 else (np.log if base == np.e else (lambda x: np.log(x)/np.log(base)))
        SvN.append(float(-np.sum(nz * log_fn(nz))))
    return SvN

# ---------- Curves Plots ----------

def plot_purity(pur_list, dt=1.0, title="Purity Tr(ρ²)(t)") -> None:
    t = np.arange(len(pur_list))*dt
    plt.figure(figsize=(6,3.2))
    plt.plot(t, pur_list, lw=2); plt.ylim(0,1.01); plt.grid(alpha=.25)
    plt.xlabel("Time"); plt.ylabel("Purity Tr(ρ²)"); plt.title(title); plt.show()

def plot_entropy_lists(S_list, dt=1.0, title="Von Neumann entropy S(ρ)(t)") -> None:
    t = np.arange(len(S_list))*dt
    plt.figure(figsize=(6,3.2))
    plt.plot(t, S_list, lw=2)   
    plt.ylim(0, None); plt.grid(alpha=.25)
    plt.xlabel("Time"); plt.ylabel("S(ρ)")
    plt.title(title); plt.show()

def plot_concurrence(C_list, dt=1.0, title="Wootters concurrence C(t)") -> None:
    t = np.arange(len(C_list))*dt
    plt.figure(figsize=(6,3.2))
    plt.plot(t, C_list, lw=2)
    plt.ylim(0, 1.02); plt.grid(alpha=.25)
    plt.xlabel("Time"); plt.ylabel("C")
    plt.title(title); plt.show()



####### Continuous evolution ########

def evolve_unitary_2q(rho2: DensityMatrix, U4: Operator) -> DensityMatrix:
    M = np.asarray(rho2.data, dtype=complex)
    U = np.asarray(U4.data, dtype=complex)
    M = U @ M @ U.conj().T
    return DensityMatrix(M)

def continuous_path_2q(
    rho0_2q: DensityMatrix,
    ops_2q: Iterable[Operator],
    steps_per_gate: int = 60,
    *,
    kraus_channels: Optional[List[Iterable[np.ndarray]]] = None,
    noise_start: int = 0,
    noise_end: Optional[int] = None,
    return_rhos: bool = False,
)-> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]], List[float]]:
    """
    Continuous 2-qubits evolution:
      - Each 4x4 gate is cut into 'steps_per_gate' gates thanks to logm/expm.
      - For each step, if the gate index belongs to [noise_start, noise_end], we apply
        (in the order) all of the specified noise channels on A and/or B (lifted in 4x4).

    Returns:
      ptsA: List[(x,y,z)] for rho_A(t)
      ptsB: List[(x,y,z)] for rho_B(t)
      purG: List[float]   purity of rho_AB(t)
    """
    ops = list(ops_2q)
    nG = len(ops)
    if nG == 0:
        rhoA, rhoB = reduced_states_from_2q(rho0_2q)
        out = ([bloch_vector_rho_1q(rhoA)], [bloch_vector_rho_1q(rhoB)], [purity(rho0_2q)])
        if return_rhos:
            return (*out, [rho0_2q])

    if noise_end is None:
        noise_end = nG - 1
    noise_start = max(0, int(noise_start))
    noise_end   = min(nG - 1, int(noise_end))
    steps_per_gate = max(1, int(steps_per_gate))

    has_noise = kraus_channels is not None and len(kraus_channels) > 0
   

    rho = rho0_2q
    rhoA, rhoB = reduced_states_from_2q(rho)
    ptsA = [bloch_vector_rho_1q(rhoA)]
    ptsB = [bloch_vector_rho_1q(rhoB)]
    purG = [purity(rho)]
    rhos_hist = [rho] if return_rhos else None

    for gate_idx, U in enumerate(ops):
        # Unitary step cut for that gate
        U_step = unitary_step(U, steps_per_gate)

        for _ in range(steps_per_gate):
            # 1) Unitary application
            rho = evolve_unitary_2q(rho, U_step)

            # 2) Noise application (if the gate index is in the noise window)
            if noise_start <= gate_idx <= noise_end:
                if has_noise:
                    for Kset in kraus_channels:
                        rho = apply_kraus_2q(rho, Kset)

            # 3) Samplings
            rhoA, rhoB = reduced_states_from_2q(rho)
            ptsA.append(bloch_vector_rho_1q(rhoA))
            ptsB.append(bloch_vector_rho_1q(rhoB))
            purG.append(purity(rho))
            if return_rhos:
                rhos_hist.append(rho)

    if return_rhos:
        return ptsA, ptsB, purG, rhos_hist
    return ptsA, ptsB, purG


####### Animation #######

def animate_trajectory(points_A, points_B, interval_ms=25, show_trail=True) -> Tuple[FuncAnimation, FuncAnimation]:
    if not points_A or not points_B:
        raise ValueError("A list is empty.")
    pts_A = np.asarray(points_A, dtype=float)
    pts_B = np.asarray(points_B, dtype=float)

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
    scat_A = ax.scatter([pts_A[0,0]], [pts_A[0,1]], [pts_A[0,2]], s=50, c="red")
    scat_B = ax.scatter([pts_B[0,0]], [pts_B[0,1]], [pts_B[0,2]], s=50, c="blue")
    line_A, = ax.plot([], [], [], linewidth=1.5, alpha=0.7)
    line_B, = ax.plot([], [], [], linewidth=1.5, alpha=0.7)

    def update_A(i):
        x,y,z = pts_A[i]
        scat_A._offsets3d = ([x], [y], [z])
        if show_trail:
            Xs, Ys, Zs = pts_A[:i+1,0], pts_A[:i+1,1], pts_A[:i+1,2]
            line_A.set_data(Xs, Ys); line_A.set_3d_properties(Zs)
        ax.set_title(f"Frame {i+1}/{len(pts_A)}")
        return scat_A, line_A
    
    def update_B(i):
        x,y,z = pts_B[i]
        scat_B._offsets3d = ([x], [y], [z])
        if show_trail:
            Xs, Ys, Zs = pts_B[:i+1,0], pts_B[:i+1,1], pts_B[:i+1,2]
            line_B.set_data(Xs, Ys); line_B.set_3d_properties(Zs)
        ax.set_title(f"Frame {i+1}/{len(pts_B)}")
        return scat_B, line_B

    anim_A = FuncAnimation(fig, update_A, frames=len(pts_A), interval=interval_ms, blit=False, repeat=True)
    anim_B = FuncAnimation(fig, update_B, frames=len(pts_B), interval=interval_ms, blit=False, repeat=True)
    plt.show()
    return anim_A, anim_B


####### tests #######

ket00 = Statevector([1, 0, 0, 0])
ket01 = Statevector([0, 1, 0, 0])
ket10 = Statevector([0, 0, 1, 0])
ket11 = Statevector([0, 0, 0, 1])

phip = Statevector(1/np.sqrt(2)*np.array([1,0,0,1]))
phim = Statevector(1/np.sqrt(2)*np.array([1,0,0,-1]))
psip = Statevector(1/np.sqrt(2)*np.array([0,1,1,0]))
psim = Statevector(1/np.sqrt(2)*np.array([0,1,-1,0]))

d00 = DensityMatrix(ket00)
d01 = DensityMatrix(ket01)
d10 = DensityMatrix(ket10)
d11 = DensityMatrix(ket11)

dphip = DensityMatrix(phip)
dphim = DensityMatrix(phim)
dpsip = DensityMatrix(psip)
dpsim = DensityMatrix(psim)


seq_A = [H1]
seq_B = [I1]
seq = embed2(seq_B, seq_A)
seq.append(CX)
seq += embed2(seq_B,seq_A)


n_seq = len(seq)
steps_per_gate = 90
total_steps = n_seq * steps_per_gate
dt = 0.001


t_noise_steps = total_noise_steps(0,n_seq,steps_per_gate = steps_per_gate, n_ops = n_seq)
t_noise = t_noise_steps * dt # Noise duration


kraus_dep_discret = discreet_depolarizing(0.5, total_steps=t_noise_steps)
kraus_T1_discret  = discreet_T1(t_noise, 0.4, total_steps=t_noise_steps)
kraus_T2_discret  = discreet_T2(t_noise, 0.2, total_steps=t_noise_steps)


kraus_noise_A = [kraus_T1_discret]
kraus_noise_B = [[I1,I1]]
kraus_channels = [embed_kraus(A, B) for A,B in zip(kraus_noise_A, kraus_noise_B)]


pts0_A, pts0_B, purity0_global, rhos0 = continuous_path_2q(d00, seq, steps_per_gate=steps_per_gate, return_rhos=True)
pts_A,  pts_B,  purity_global,  rhos  = continuous_path_2q(d00, seq, steps_per_gate=steps_per_gate, kraus_channels=kraus_channels, noise_start=0, noise_end=n_seq, return_rhos=True)

S0_A, S0_B = entropy_vN(pts0_A, base=2.0), entropy_vN(pts0_B, base=2.0)
S_A,  S_B  = entropy_vN(pts_A, base=2.0), entropy_vN(pts_B, base=2.0)

C0 = [concurrence_2q(r) for r in rhos0]
C  = [concurrence_2q(r) for r in rhos]

radius(pts0_A, title = "A radius without noise")
radius(pts0_B, title = "B radius without noise")
radius(pts_A, title = "A radius with noise")
radius(pts_B, title = "B radius with noise")

plot_purity(purity0_global, title = "System purity without noise")
plot_purity(purity_global, title = "System purity with noise")

plot_entropy_lists(S0_A, dt, title="S(ρ_A) without noise")
plot_entropy_lists(S0_B, dt, title="S(ρ_B) without noise")
plot_entropy_lists(S_A,  dt, title="S(ρ_A) with noise")
plot_entropy_lists(S_B,  dt, title="S(ρ_B) with noise")

plot_concurrence(C0, dt, title="Concurrence C(t) without noise")
plot_concurrence(C,  dt, title="Concurrence C(t) with noise")

anim0 = animate_trajectory(pts0_A, pts0_B, interval_ms=20)
anim = animate_trajectory(pts_A, pts_B, interval_ms = 20)