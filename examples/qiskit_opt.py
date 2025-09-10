from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler import PassManager
import sys

from qiskit.qasm2 import dump


def save_qiskit_as_qasm(
    circuit: QuantumCircuit, output_path: str = "data/output.qasm"
) -> None:
    f = open(output_path, "w")

    dump(circuit, f)


def optimize_qasm(qasm_path: str, output_path: str):

    circuit = QuantumCircuit.from_qasm_file(qasm_path)
    print(f"Original depth: {circuit.depth()}")

    optimized = transpile(circuit, optimization_level=3, basis_gates=["cz", "rx", "rz"])
    print(f"Optimized depth (level 3): {optimized.depth()}")

    # Save the optimized circuit
    save_qiskit_as_qasm(optimized, output_path=output_path)

    print(f"Optimized QASM saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python optimize_qasm.py input.qasm output.qasm")
    else:
        optimize_qasm(sys.argv[1], sys.argv[2])
