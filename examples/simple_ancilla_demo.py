#!/usr/bin/env python3

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from extract_unitaries import create_ancilla_circuit

def create_simple_test_unitary():
    """Create a simple test unitary for demonstration"""
    # Simple 2-qubit unitary (4x4 matrix)
    U = np.array([
        [1, 0, 0, 0],
        [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
        [0, 1/np.sqrt(2), -1/np.sqrt(2), 0],
        [0, 0, 0, 1]
    ], dtype=complex)
    return U

def demonstrate_ancilla_in_lindbladian():
    print("=" * 60)
    print("ANCILLA CIRCUIT DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple quantum circuit with ancilla
    num_qubits = 3
    qc = QuantumCircuit(num_qubits, 1)
    
    print("1. Initial circuit with some quantum operations:")
    qc.h(1)
    qc.cz(0, 1)
    qc.rz(np.pi/4, 2)
    print(qc.draw())
    print()
    
    print("2. Ancilla reset circuit (measure + reset qubit 0):")
    ancilla_reset = create_ancilla_circuit(num_qubits, ancilla_qubit_idx=0)
    print(ancilla_reset.draw())
    print()
    
    print("3. Combined circuit - Evolution + Ancilla Reset:")
    combined = qc.compose(ancilla_reset)
    print(combined.draw())
    print()
    
    print("4. Simulating multiple evolution steps with periodic resets:")
    full_circuit = QuantumCircuit(num_qubits, 1)
    
    # Evolution step 1
    test_unitary = create_simple_test_unitary()
    if test_unitary.shape[0] <= 2**num_qubits:
        # Pad to full size if needed
        if test_unitary.shape[0] < 2**num_qubits:
            full_unitary = np.eye(2**num_qubits, dtype=complex)
            full_unitary[:test_unitary.shape[0], :test_unitary.shape[1]] = test_unitary
        else:
            full_unitary = test_unitary[:2**num_qubits, :2**num_qubits]
        
        unitary_gate = UnitaryGate(full_unitary)
        full_circuit.append(unitary_gate, list(range(num_qubits)))
    
    # Reset after 2 steps (simulating your counter % 2 == 0 logic)
    reset_circuit = create_ancilla_circuit(num_qubits, 0)
    full_circuit = full_circuit.compose(reset_circuit)
    
    # Evolution step 2
    if test_unitary.shape[0] <= 2**num_qubits:
        full_circuit.append(unitary_gate, list(range(num_qubits)))
    
    print("Complete Lindbladian simulation circuit:")
    print(full_circuit.draw())
    print()
    
    print("=" * 60)
    print("IMPLEMENTATION NOTES:")
    print("=" * 60)
    print("• create_ancilla_circuit(n_qubits, idx) creates measure + reset")
    print("• Use .compose() to combine circuits sequentially")
    print("• Reset frequency matches your counter % 2 == 0 logic")
    print("• Ancilla qubit 0 corresponds to your first qubit")
    print("• This simulates the 'fresh ancilla' from your matrix code")

if __name__ == "__main__":
    demonstrate_ancilla_in_lindbladian()
