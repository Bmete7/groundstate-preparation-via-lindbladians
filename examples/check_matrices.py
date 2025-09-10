import numpy as np
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import sys
import os

from generate_path import (
    generate_all_pickled_K,
    generate_fidelity_plot_path,
    generate_psi_0_path,
    generate_psi_GS_path,
    generate_qasm_path,
    generate_bqskit_output_path,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extract_unitaries import (
    Unitary,
    generate_qiskit_circuit_from_unitary,
    transpile_qiskit,
    compile_circuit,
    save_qiskit_as_qasm,
    calculate_approximation_error,
    reverse_qubit_order,
    phase_corrected_frobenius_norm,
)

L = 2

from utils import fresh_ancilla_rho, partial_trace

basis_gates = [
    "rx",
    "rz",
    "cz",
    "swap",
]

all_matrices = np.load(generate_all_pickled_K(L), allow_pickle=True)

psi_0 = np.load(generate_psi_0_path(L), allow_pickle=True) + 0j
psi_GS = (
    np.load(generate_psi_GS_path(L), allow_pickle=True) + 0j
)  # Make them complex by default

# psi_0 = np.zeros_like(psi_0, dtype=complex)
# psi_0[0] = 1.0


overlaps = []
# overlaps.append(np.abs(np.vdot(psi_GS, psi_0)))
# Initialize state
n_qubits = int(np.log2(all_matrices[0].shape[0]))  # also given by L
# rho = np.array([[1, 0], [0, 0]], dtype=complex)
# for _ in range(n_qubits - 1):
#     rho = np.kron(rho, np.array([[1, 0], [0, 0]], dtype=complex))  # |00...0><00...0|
rho = np.outer(psi_0, psi_0.conj()) + 0j  # Initial state in density matrix formalism
rho = np.kron(np.array([[1, 0], [0, 0]]), rho)  # |00...0><00...0|
# Apply in reverse order


counter = 0
for mat in all_matrices:
    # Evolve: ρ → U ρ U†
    rho = mat @ rho @ mat.conj().T

    # After every 2 steps, do partial trace + re-add ancilla
    if (
        counter % 2 == 0 and counter > 0
    ):  # skip the first step as it is the general hamiltonian evolution
        dim = int(rho.shape[0])

        dimA, dimB = 2, dim // 2  # Assume first qubit has dim 2
        rhoA, rhoB = partial_trace(rho, dimA, dimB)
        fidelity = np.real(psi_GS.conj().T @ rhoB @ psi_GS)  # Fidelity computation
        overlaps.append(fidelity)
        rho = fresh_ancilla_rho(rhoB)  # Reinsert ancilla as |0><0| ⊗ rhoB
    counter += 1

# Final state
final_rho = rho

print(f"Final fidelity with ground state: {overlaps[-1]}")
plt.figure(figsize=(8, 5))
plt.plot(overlaps, marker="o", linestyle="-", color="navy")
plt.xlabel("Partial Trace Step")
plt.ylabel("Fidelity with Ground State")
plt.title("Overlap (Fidelity) After Each Partial Trace")
plt.grid(True)
plt.tight_layout()
plt.savefig(generate_fidelity_plot_path(), dpi=150)  # Optional: save the figure
plt.show()


def main(L: int):

    K_tilde_path = generate_all_pickled_K(L)
    qasm_path = generate_qasm_path(L)
    bqskit_output_path = generate_bqskit_output_path(L)
    qasm_generated = False
    print(f"Processing {K_tilde_path}")
    U = Unitary(K_tilde_path)
    print(f" len {len(U.U)}")
    print(f"Number of qubits: {U.U[0].shape}")

    circuit = generate_qiskit_circuit_from_unitary(U)
    circuit = transpile_qiskit(circuit, basis_gates)
    if not qasm_generated:
        save_qiskit_as_qasm(circuit, qasm_path)
    circuit, compiled_circuit = compile_circuit(
        qasm_path, bqskit_output_path, verbose=True
    )
    original_U = circuit.get_unitary().numpy
    compiled_U = (compiled_circuit.get_unitary()).numpy

    # Sometimes there is a global phase -1 on the synthesised unitary. Check if that is the case
    approx_error = calculate_approximation_error(
        original_U, compiled_U, phase_corrected_frobenius_norm
    )
    original_U_bqskit_corrected = reverse_qubit_order(original_U)
    compiled_U_bqskit_corrected = reverse_qubit_order(compiled_U)

    approx_error_compiled = calculate_approximation_error(
        U.U, compiled_U_bqskit_corrected, phase_corrected_frobenius_norm
    )

    approx_error_original = calculate_approximation_error(
        original_U_bqskit_corrected, U.U, phase_corrected_frobenius_norm
    )

    print(f"Synthesis approximation error: {approx_error}")
    print(f"Original approximation error: {approx_error_original}")
    print(f"Compiled approximation error: {approx_error_compiled}")
    star = "*"

    print(f"{star * 50}")
    # print(f"Original Unitary: {original_U_bqskit_corrected}")
    # print(f"Compiled Unitary: {compiled_U_bqskit_corrected}")

    print(compiled_U_bqskit_corrected @ np.kron(psi_0, np.array([1, 0])))
    resulting_psi = compiled_U_bqskit_corrected @ np.kron(psi_0, np.array([1, 0]))
    resulting_rho = resulting_psi.reshape(-1, 1) @ resulting_psi.reshape(-1, 1).conj().T
    reduced_resulting_psi = partial_trace(resulting_rho, 2, dim // 2)[1]
    psi_GS.conj().T @ reduced_resulting_psi @ psi_GS


if __name__ == "__main__":
    L = 2
    print(f"Running main with L={L}")
    main(L)
