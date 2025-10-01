Bloch Simulator — 1 & 2 Qubits

A Python-based simulator for visualizing quantum state evolution on the Bloch sphere, designed for one- and two-qubit systems.
It allows continuous unitary evolution, application of Kraus noise channels (T1, T2, depolarizing), and extraction of physical metrics such as purity, von Neumann entropy, concurrence (2 qubits), and fidelity between trajectories.

This project was developed as part of my self-training in quantum information and aims to provide both an educational tool and a research-oriented prototype.
It is particularly relevant to my applications for internships in quantum computing & quantum information theory (e.g., IBM Quantum, research labs, academic programs).

✨ Features

1 Qubit Simulator

Continuous unitary evolution via axis-angle decomposition

Application of noise models (T1, T2, depolarizing)

Visualization of trajectories inside the Bloch ball

Metrics: radius, fidelity between noisy and noiseless trajectories

2 Qubit Simulator

Continuous evolution with local and non-local gates (CX, SWAP, etc.)

Noise embedding on individual qubits via Kraus operators

Reduced states & Bloch representation for each qubit

Metrics: global purity, local entropies, concurrence

Animation of qubits’ Bloch trajectories

📊 Examples

Noiseless evolution: unitary trajectories on the Bloch sphere

Effect of noise: gradual shrinking of Bloch vectors

Entanglement analysis: von Neumann entropy of reduced states and concurrence plots

Animations and plots are generated automatically using Matplotlib.

⚙️ Installation

Clone the repository:

git clone https://github.com/lettynail/bloch_simulator.git
cd bloch_simulator


Create a virtual environment and install dependencies:

python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt

🚀 Usage

Run 1-qubit simulation:

python bloch_simulator_1_qbit.py


Run 2-qubit simulation:

python bloch_simulator_2_qbits.py


Both scripts generate:

Bloch sphere trajectories

Radius and purity plots

Concurrence/entropy curves (2 qubits)

Animations

📚 Background

The project was built to deepen my expertise in:

Quantum information theory (entanglement, decoherence, density matrices)

Simulation of open quantum systems (via Kraus maps)

Visualization techniques for educational and research purposes

It complements my academic training in quantum computing engineering, as well as certifications from IBM Quantum (Qiskit, VQE, quantum algorithms).

📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

🙋‍♂️ Author

LETTY Naïl
Engineering student at ISAE-Supaero (France)
Focused on quantum computing, quantum information, and quantum simulation
Currently seeking a research internship (Fall 2026) in quantum computing

Feel free to connect via LinkedIn or check out my other projects.
