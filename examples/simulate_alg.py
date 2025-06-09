import pennylane as qml
import numpy as np
import pickle

file_path = "data/TFIM2GS.pickle"
with open(file_path, "rb") as f:
    gs_ket = pickle.load(f)

with open("data/bqskit2.qasm", "r") as f:
    qasm_str = f.read()

qc = qml.from_qasm(qasm_str)

ansatz = qml.from_qasm(qasm_str)

dev = qml.device("default.qubit", 3)
dev_repeated = qml.device("default.qubit", 3)
# measurements = [qml.var(qml.Z(0))]

circuit = qml.QNode(qml.from_qasm(qasm_str, measurements=qml.state()), dev)


state = circuit()


# TODO1; Run the full algorithm with eîth (the full circuit)
# TODO2: Find the Approx. error for bqskt
# TODO3 IDEA : Keep Fresh ancillas within the rydberg radius, only swapping when queue is over, so you shuttle them in parallel with less overhead
# TODO: This requires a complete NA compilation tool
# TODO: Contact Johannes Zeiher send the circuit and let them know about the details
# TODO: Another ideas are, multi qubit gates instead of two
# TODO: Long range interactions with multi qubit sites instead of swapping
@qml.qnode(dev_repeated)
def circuit_repeated():
    ansatz()
    ansatz()
    return qml.state()


state_repeated = circuit_repeated()


def project_onto_zero(state, target_wire, num_wires):
    """Collapse state onto |0> at target_wire, return renormalized statevector."""
    # Build projector |0⟩⟨0| on the full system
    P0 = np.array([[1, 0], [0, 0]])
    eye = [np.eye(2)] * num_wires
    eye[target_wire] = P0
    projector = eye[0]
    for op in eye[1:]:
        projector = np.kron(projector, op)

    # Apply projector
    collapsed_state = projector @ state
    norm = np.linalg.norm(collapsed_state)
    if norm < 1e-10:
        raise ValueError("Post-selection probability is ~0.")
    return collapsed_state / norm


reduced_state = project_onto_zero(state, 0, 3)
reduced_state = reduced_state[: int(len(reduced_state) / 2)]


reduced_state_repeated = project_onto_zero(state_repeated, 0, 3)
reduced_state_repeated = reduced_state_repeated[: int(len(reduced_state_repeated) / 2)]


def inner_product(psi1, psi2):
    return np.abs((psi1.conj().T @ psi2) ** 2)


psi0 = np.array([1, 0, 0, 0])

print(f"Inner product with initial state is: {inner_product(gs_ket, psi0)}")
print(f"Inner product with final state is: {inner_product(gs_ket, reduced_state)}")
print(
    f"Inner product with final state with repeated K Tilde is: {inner_product(gs_ket, reduced_state_repeated)}"
)
