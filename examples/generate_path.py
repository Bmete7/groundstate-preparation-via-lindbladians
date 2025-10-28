import datetime
import os


def return_current_time():
    """
    Returns the current time as a string formatted as 'YYYY-MM-DD_HH-MM-SS'.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d")


def generate_all_pickled_K(data_path: str):
    """Given the system size L, generate the path to save all pickled K_tilde matrices.

    Args:
        L (int): The system size.
        reps (int): The number of repetitions.
        data_path (str): The root data path to save the file.
    """
    path = f"{data_path}/dilated_unitaries.pickle"
    return os.path.join(path)


def generate_qasm_path(data_path: str):
    """Given the system size L, generate the path to save the QASM file.

    Args:
        L (int): The system size.
    """

    path = f"{data_path}/output_original_circuit.qasm"
    return os.path.join(path)


def generate_bqskit_output_path(data_path: str):
    """Given the system size L, generate the path to save the BQSkIT output file.

    Args:
        L (int): The system size.
    """

    path = f"{data_path}/bqskit_circuit.qasm"
    return os.path.join(path)


def generate_fidelity_plot_path(data_path: str):
    """Generate the path to save the fidelity plot.

    Args:
        L (int): The system size.
    """
    path = f"{data_path}/fidelity_plot.png"
    return os.path.join(path)


def generate_psi_0_path(data_path: str):
    """Given the system size L, generate the path to save the initial state psi_0.

    Args:
        L (int): The system size.
    """
    path = f"{data_path}/psi0.npy"
    return os.path.join(path)


def generate_psi_GS_path(data_path: str):
    """Given the system size L, generate the path to save the ground state psi_GS.

    Args:
        L (int): The system size.
    """
    path = f"{data_path}/psi_GS.npy"
    return os.path.join(path)


def generate_experiment_config(data_path: str):
    """Generate the path to save the experiment configuration file."""
    path = f"{data_path}/experiment.json"
    return os.path.join(path)
