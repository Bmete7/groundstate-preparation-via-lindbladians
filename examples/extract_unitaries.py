from bqskit import compile, Circuit
import numpy as np
from qiskit import QuantumCircuit
import pickle
import time
from typing import Callable, Any
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

basis_gates = [
    "rx",
    "rz",
    "cz",
    "swap",
]


def hilbert_schmidt(U, V):
    d = U.shape[0]
    return np.abs(np.trace(np.conj(U.T) @ V)) / d


def average_gate_infidelity(U, V):
    d = U.shape[0]
    fidelity = np.abs(np.trace(np.conj(U.T) @ V)) ** 2 / d**2
    return 1 - (d / (d + 1)) * (1 - fidelity)


def phase_corrected_frobenius_norm(U, V, ord="fro"):
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
    return np.linalg.norm(U, V_phase_corrected, ord=ord)


def calculate_approximation_error(U, V, callback):
    """Given a bqskit synthesised unitary and a general unitary, find the approximation error

    Args:
        U (np.ndarray): First unitary
        V (np.ndarray): Second unitary

    Returns:
        float: approximation error
    """
    return callback(U, V)


class Unitary:
    _num_qubits = None
    _num_params = 0
    U = None
    _name = "dilated_expK"
    _qasm_name = "dilated_expK"

    def __init__(self, file_path: str):
        self.U = self.load_unitary(file_path)
        self._num_qubits = int(math.log2(self.U.shape[0]))

    def load_unitary(self, file_path: str) -> np.ndarray:
        """Given a file_path as a pickle, save it as a matrix

        Args:
            file_path (str): Input file path

        Returns:
            np.ndarray: Unitary as matrix
        """

        with open(file_path, "rb") as f:
            U = pickle.load(f)  # get all the quantum gates in a list
        return U

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def num_params(self):
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

    assert U.U.shape[0] == 2**qubits
    qc = QuantumCircuit(qubits, qubits)
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
    circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=3)
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

    # workflow = [
    #     QSearchSynthesisPass(),
    # ]
    # with Compiler() as compiler:
    #     synthesized_circuit = compiler.compile(circuit, workflow)

    # if verbose:
    #     print(f"-----------------------------------")
    #     print(f"Gate counts after BQSKit Optimization")
    #     for gate in synthesized_circuit.gate_set:
    #         print(f"{gate} Count:", synthesized_circuit.count(gate))
    #     print(f"-----------------------------------")
    # input("Succeeded")
    return circuit, compiled_circuit


for i in range(2, 4):
    K_tilde_path = "data/TFIM" + str(i) + "_KTilde.pickle"
    qasm_path = "data/output" + str(i) + ".qasm"
    bqskit_output_path = "data/bqskit" + str(i) + ".qasm"
    qasm_generated = False
    print(f"Processing {K_tilde_path}")
    U = Unitary(K_tilde_path)
    # print(U)
    # dilated_K_file_path = "data/dilated_expK.pickle"
    # qasm_path = "data/output.qasm"
    # bqskit_output_path = "data/bqskit.qasm"

    circuit = generate_qiskit_circuit_from_unitary(U)
    circuit = transpile_qiskit(circuit, basis_gates)
    if not qasm_generated:
        save_qiskit_as_qasm(circuit, qasm_path)
    circuit, compiled_circuit = compile_circuit(
        qasm_path, bqskit_output_path, verbose=True
    )
    original_U = circuit.get_unitary()
    compiled_U = compiled_circuit.get_unitary()
    approx_error = calculate_approximation_error(
        original_U, compiled_U, phase_corrected_frobenius_norm
    )
    print(f"Synthesis approximation error: {approx_error}")
    star = "*"

    print(original_U - compiled_U)
    print(f"{star * 50}")
