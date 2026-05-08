import numpy as np
import matplotlib.pyplot as plt
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD
# Import the GroundStateEigensolver specifically
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
# We use the standard Estimator which is compatible with the latest Qiskit
from qiskit.primitives import Estimator
from qiskit.primitives import Estimator as QiskitEstimator # V2 Estimator
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session

# --- 1. IBM CLOUD AUTHENTICATION ---
service = QiskitRuntimeService(channel="ibm_quantum", token=" ")
backend = service.backend("ibm_marrakesh") # Use a utility-scale 127-qubit system

# --- 2. THE PROCA EXPERIMENT FUNCTION ---
def run_proca_simulation(mu_val):
    # Driver for H2 proxy as per Chapter 2 of your thesis
    driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g")
    problem = driver.run()

    # Modify integrals using the Yukawa Factor: e^(-mu * r)[cite: 1, 3]
    proca_factor = np.exp(-mu_val * 0.735) 
    problem.hamiltonian.electronic_integrals.alpha_beta *= proca_factor

    mapper = JordanWignerMapper()
    ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper)
    
    # Standard Estimator for legacy compatibility
    estimator = Estimator() 
    vqe = VQE(estimator, ansatz, SLSQP(maxiter=25))
    calc = GroundStateEigensolver(mapper, vqe)
    
    result = calc.solve(problem)
    return result.total_energies[0]

# --- 3. DATA COLLECTION (PHENOMENOLOGY) ---
mass_range = [0.0, 0.05, 0.1, 0.15] # Testing different 'mu' values
energies = []

print("Starting Multi-Mass Proca Simulation...")
for mu in mass_range:
    energy = run_proca_simulation(mu)
    energies.append(energy)
    print(f"Mu: {mu} | Energy: {energy:.6f} Hartree")

# --- 4. VISUALIZE THE "PHASE BINDING" ---
plt.plot(mass_range, energies, marker='o', color='#003366')
plt.title("Impact of Photon Mass (Proca) on Molecular Ground State")
plt.xlabel("Photon Mass (mu)")
plt.ylabel("Ground State Energy (Hartree)")
plt.grid(True)
plt.show()