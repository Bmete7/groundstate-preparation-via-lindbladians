#!/usr/bin/env python3

import numpy as np
from qiskit import QuantumCircuit
from extract_unitaries import (
    create_ancilla_circuit,
    generate_full_lindbladian_circuit,
    Unitary,
)


def test_basic_ancilla_reset():
    print("Testing basic ancilla reset circuit:")

    num_qubits = 4
    ancilla_circuit = create_ancilla_circuit(num_qubits, ancilla_qubit_idx=0)

    print(f"Ancilla reset circuit for {num_qubits} qubits:")
    print(ancilla_circuit.draw())
    print()


def test_lindbladian_with_resets():
    print("Testing full Lindbladian circuit with periodic resets:")

    K_tilde_path = "data/TFIM3_ALL_PICKLED_K_2025-08-01.pickle"
    U = Unitary(K_tilde_path)

    lindbladian_circuit = generate_full_lindbladian_circuit(U, reset_frequency=2)

    print(f"Lindbladian circuit with {U.num_qubits} qubits:")
    print(f"Total circuit depth: {lindbladian_circuit.depth()}")
    print(f"Number of operations: {len(lindbladian_circuit)}")
    print()


def demonstrate_ancilla_usage():
    print("Demonstrating ancilla reset usage in quantum simulation:")
    print("1. Apply unitary evolution")
    print("2. Measure ancilla qubit")
    print("3. Reset ancilla to |0⟩")
    print("4. Continue with fresh ancilla")
    print()

    qc = QuantumCircuit(3, 1)

    qc.h(1)
    qc.cz(0, 1)
    qc.rz(np.pi / 4, 2)

    print("Step 1 - Apply evolution:")
    print(qc.draw())
    print()

    ancilla_reset = create_ancilla_circuit(3, 0)
    full_circuit = qc.compose(ancilla_reset)

    qc.h(2)
    qc.cz(1, 2)

    full_circuit = full_circuit.compose(qc)

    print("Step 2 - With ancilla reset:")
    print(full_circuit.draw())


if __name__ == "__main__":
    test_basic_ancilla_reset()
    test_lindbladian_with_resets()
    demonstrate_ancilla_usage()
