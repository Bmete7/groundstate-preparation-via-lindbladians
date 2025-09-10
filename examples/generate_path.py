import datetime
import os


def return_current_time():
    """
    Returns the current time as a string formatted as 'YYYY-MM-DD_HH-MM-SS'.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d")


def generate_all_pickled_K(L: int):
    """Given the system size L, generate the path to save all pickled K_tilde matrices.

    Args:
        L (int): The system size.
    """
    cwd = os.getcwd()  # get current working directory

    path = f"data/TFIM_{L}_ALL_PICKLED_K_{return_current_time()}.pickle"
    return os.path.join(cwd, path)


def generate_qasm_path(L: int):
    """Given the system size L, generate the path to save the QASM file.

    Args:
        L (int): The system size.
    """

    cwd = os.getcwd()  # get current working directory

    path = f"data/output{L}_qubits.qasm"
    return os.path.join(cwd, path)


def generate_bqskit_output_path(L: int):
    """Given the system size L, generate the path to save the BQSkIT output file.

    Args:
        L (int): The system size.
    """
    cwd = os.getcwd()  # get current working directory

    path = f"data/bqskit{L}_qubits.qasm"
    return os.path.join(cwd, path)


def generate_fidelity_plot_path():
    """Generate the path to save the fidelity plot.

    Args:
        L (int): The system size.
    """
    cwd = os.getcwd()  # get current working directory

    path = "data/fidelity_plot.png"
    return os.path.join(cwd, path)


def generate_psi_0_path(L: int):
    """Given the system size L, generate the path to save the initial state psi_0.

    Args:
        L (int): The system size.
    """
    cwd = os.getcwd()  # get current working directory

    path = f"data/TFIM_{L}_psi0_{return_current_time()}.npy"
    return os.path.join(cwd, path)


def generate_psi_GS_path(L: int):
    """Given the system size L, generate the path to save the ground state psi_GS.

    Args:
        L (int): The system size.
    """
    cwd = os.getcwd()  # get current working directory

    path = f"data/TFIM_{L}_psi_GS_{return_current_time()}.npy"
    return os.path.join(cwd, path)
