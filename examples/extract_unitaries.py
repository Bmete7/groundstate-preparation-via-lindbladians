from bqskit import compile, Circuit
import numpy as np
from qiskit import QuantumCircuit
import pickle
import time
from typing import Callable, Any, Tuple
from functools import wraps
import math
from qiskit.circuit.library import UnitaryGate
from qiskit.qasm2 import dump
from qiskit import transpile
from bqskit.compiler.gateset import GateSet
from bqskit.ir.gates import RXGate, RZGate, CZGate, SwapGate
from bqskit import MachineModel
from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.passes import (
    QuickPartitioner,
    ForEachBlockPass,
    ForEachBlockPass,
    QSearchSynthesisPass,
    QSearchSynthesisPass,
    ScanningGateRemovalPass,
    UnfoldPass,
    GeneralSQDecomposition,
)
from functools import reduce


def hilbert_schmidt(U, V):
    d = U.shape[0]
    return np.abs(np.trace(np.conj(U.T) @ V)) / d


def average_gate_infidelity(U, V):
    d = U.shape[0]
    fidelity = np.abs(np.trace(np.conj(U.T) @ V)) ** 2 / d**2
    return 1 - (d / (d + 1)) * (1 - fidelity)


def phase_corrected_frobenius_norm(U: np.ndarray, V: np.ndarray, ord="fro"):
    """Given 2 matrices, calculate if there is a global phase difference (i.e. -1), and then calculate the norm of the difference between two operators

    Args:
        U (np.ndarray): 1st matrix
        V (np.ndarray): 2nd matrix
        ord (str, optional): Operator norm order. Defaults to 'fro'.

        Returns:
            float: Approximation error
    """
    phase = np.angle(np.trace(np.dot(np.conj(U.T), V)))
    V_phase_corrected = V * np.exp(-1j * phase)
    return np.linalg.norm(U - V_phase_corrected, ord=ord)


def calculate_approximation_error(U, V, callback):
    """Given a bqskit synthesised unitary and a general unitary, find the approximation error

    Args:
        U (np.ndarray): First unitary
        V (np.ndarray): Second unitary

    Returns:
        float: approximation error
    """
    return callback(U, V)


def reverse_qubit_order(U: np.ndarray) -> np.ndarray:
    """Reverse the qubit order in a unitary matrix using NumPy.

    Args:
        U: A (2^n x 2^n) unitary matrix.

    Returns:
        A new unitary matrix with reversed qubit ordering.
    """
    dim = U.shape[0]
    n = int(np.log2(dim))
    assert 2**n == dim, "Input must be a 2^n x 2^n matrix"

    # Reshape into tensor: (2, 2, ..., 2) x (2, 2, ..., 2)
    U_tensor = U.reshape([2] * n * 2)

    # Reverse qubit order by reversing the axes
    axes = list(range(2 * n))
    perm = list(reversed(axes[:n])) + list(reversed(axes[n:]))

    U_reordered = np.transpose(U_tensor, axes=perm)

    # Flatten back into matrix
    return U_reordered.reshape(dim, dim)


class Unitary:
    _num_qubits: int = 0
    _num_params: int = 0
    U: np.ndarray = None
    _name: str = "dilated_expK"
    _qasm_name: str = "dilated_expK"

    def __init__(self, file_path: str):
        self.U, self.U_raw = self.load_unitary(file_path)

        try:
            self._num_qubits = int(math.log2(self.U.shape[0]))
        except:
            self._num_qubits = int(math.log2(self.U[0].shape[0]))

    def load_unitary(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Given a file_path as a pickle, save it as a matrix

        Args:
            file_path (str): Input file path

        Returns:
            np.ndarray: Unitary as matrix
        """

        with open(file_path, "rb") as f:
            U = pickle.load(f)  # get all the quantum gates in a list
        if type(U) == list:
            U_processed = reduce(lambda a, b: b @ a, U)  ## TODO Remove reverse
        return U_processed, U  # TODO Change this to return all the gates in a list

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_params(self) -> int:
        return self._num_params


def time_wrapper(f: Callable, *args, **kwargs):
    """Given a function as an input, time it

    Args:
        f (function): input function
    """

    def wrapped(*args, **kwargs) -> Any:

        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"Function '{f.__name__}' executed in {end - start:.4f} seconds.")
        return result

    return wrapped


def load_unitary(file_path: str) -> np.ndarray:
    """Given a file_path as a pickle, save it as a matrix

    Args:
        file_path (str): Input file path

    Returns:
        np.ndarray: Unitary as matrix
    """

    with open("", "rb") as f:
        U = pickle.load(f)  # get all the quantum gates in a list
    return U


@time_wrapper
def generate_qiskit_circuit_from_unitary(U: Unitary) -> QuantumCircuit:
    """Given an NxN unitary matrix, generate a qiskit QuantumCircuit

    Args:
        U (Unitary): np.ndarray Unitary matrix

    Returns:
        QuantumCircuit: _description_
    """
    qubits = U.num_qubits
    try:
        assert U.U.shape[0] == 2**qubits
    except:
        assert U.U[0].shape[0] == 2**qubits
    qc = QuantumCircuit(qubits, qubits)
    if type(U.U) == list:
        Ugate = [UnitaryGate(U_s) for U_s in U.U]
        for i, U_s in enumerate(Ugate):
            qc.append(U_s, list(range(0, qubits)))
    else:
        Ugate = UnitaryGate(U.U)
        qc.append(Ugate, list(range(0, qubits)))
    return qc


@time_wrapper
def transpile_qiskit(circuit: QuantumCircuit, basis_gates: list) -> QuantumCircuit:
    """Given a QuantumCircuit with a single NxN unitary matrix, produce a decomposed circuit using a fixed basis gate set.

    Args:
        circuit (QuantumCircuit): Input quantum circuit
        basis_gates (list): List of native gates

    Returns:
        QuantumCircuit: Decomposed quantum circuit
    """
    circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=1)
    return circuit


def save_qiskit_as_qasm(
    circuit: QuantumCircuit, output_path: str = "data/output.qasm"
) -> None:
    """Given a quantum circuit, save it to a qasm file

    Args:
        circuit (QuantumCircuit): Input quantum circuit
        output_path (str, optional): Path to the qasm file to save. Defaults to "data/output.qasm".
    """
    f = open(output_path, "w")
    dump(circuit, f)
    print("Qiskit qasm output saved to:", output_path)


@time_wrapper
def compile_circuit(
    input_file: str, output_file: str = "data/output.qasm", verbose: bool = True
) -> None:
    """Given an input file as a qasm file, synthesise the circuit using BQSKIT and provide a more efficient alternative

    Args:
        input_file (str): Path to the QASM Input file
        output_file (str, optional): Path to the QASM Output file for BQSKIT synthesised circuit‚. Defaults to "data/output.qasm".
        verbose (bool, optional): Prints compilation details. Defaults to True.

    Returns:
        circuit: (BQSKIT.circuit) The input circuit
        compiled_circuit: (BQSKIT.circuit) The synthesised circuit
    """
    circuit = Circuit.from_file(input_file)

    num_qubits = circuit.num_qudits
    my_basis = GateSet([RXGate(), RZGate(), CZGate(), SwapGate()])
    model = MachineModel(num_qubits, gate_set=my_basis)

    if verbose:
        print(f"-----------------------------------")
        print(f"Gate counts from the input circuit")
        for gate in circuit.gate_set:
            print(f"{gate} Count:", circuit.count(gate))
        print(f"-----------------------------------")
    compiled_circuit = compile(circuit, model=model, optimization_level=1)
    compiled_circuit.save(output_file)

    if verbose:
        print(f"-----------------------------------")
        print(f"Gate counts after BQSKit Synthesis")
        for gate in compiled_circuit.gate_set:
            print(f"{gate} Count:", compiled_circuit.count(gate))
        print(f"-----------------------------------")

    return circuit, compiled_circuit


def create_ancilla_circuit(
    qc: QuantumCircuit, ancilla_qubit_idx: int
) -> QuantumCircuit:

    return qc


@time_wrapper
def generate_full_lindbladian_circuit(
    U: Unitary, reset_frequency: int
) -> QuantumCircuit:
    if isinstance(U.U_raw, list):
        matrices = U.U_raw
    else:
        matrices = [U.U_raw]

    num_qubits = U.num_qubits + reset_frequency + 1
    qc: QuantumCircuit = QuantumCircuit(num_qubits, U.num_qubits)
    swap_ctr = 0

    for step, matrix in enumerate(matrices):
        print(f"steps: {step}")
        unitary_gate = UnitaryGate(matrix)
        qc.append(unitary_gate, list(range(U.num_qubits)))

        if (step + 1) % 2 == 0 and step > 0:
            swap_ctr += 1
            qc.measure(0, 0)
            print(f"swap_ctr: {swap_ctr}")
            print(f"swapping qubits {0} and {U.num_qubits + swap_ctr}")
            qc.swap(0, U.num_qubits + swap_ctr)
    for i in range(U.num_qubits):
        if i != 0:
            qc.measure(i, i)
    print(qc.draw())
    return qc
