import numpy as np
import json
from generate_path import generate_experiment_config


def partial_trace(
    rho: np.ndarray, dimA: int, dimB: int
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a partial trace over a bipartite state.

    Args:
        rho (np.ndarray): The density matrix to trace over.
        dimA (int): The dimension of subsystem A.
        dimB (int): The dimension of subsystem B.

    Returns:
        (np.ndarray, np.ndarray): The reduced density matrices for subsystems A and B.
    """

    rho = rho.reshape((dimA, dimB, dimA, dimB))
    rhoA = np.trace(rho, axis1=1, axis2=3)
    rhoB = np.trace(rho, axis1=0, axis2=2)
    return rhoA, rhoB


def fresh_ancilla_rho(rho: np.ndarray) -> np.ndarray:
    """Create a fresh ancilla state and combine it with the given density matrix.

    Args:
        rho (np.ndarray): The density matrix to combine with the ancilla state.

    Returns:
        np.ndarray: The combined density matrix with the ancilla state.
    """
    ancilla = np.array([[1], [0]], dtype=complex) @ np.array([[1, 0]], dtype=complex)
    return np.kron(ancilla, rho)


def load_experiment_config() -> tuple[int, int]:
    """Load experiment configuration from a JSON file.

    Args:
        path (str): The path to the JSON configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    path = generate_experiment_config()
    with open(path, "r") as f:
        config = json.load(f)
    L = config["L"]
    reps = config["reps"]
    return L, reps
