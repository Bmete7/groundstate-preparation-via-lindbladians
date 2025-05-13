import numpy as np


def crandn(size=None, rng: np.random.Generator = None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j * rng.normal(size=size)) / np.sqrt(2)


def random_density_matrix(d: int, rng: np.random.Generator):
    """
    Construct a random density matrix.
    """
    rho = crandn((d, d), rng)
    rho = rho @ rho.conj().T
    rho /= np.trace(rho)
    return rho


def partial_trace(rho, dimA, dimB):
    """
    Compute the partial traces of a density matrix 'rho' of a composite quantum system AB.

    Args:
        rho:  density matrix of dimension dimA*dimB x dimA*dimB
        dimA: dimension of subsystem A
        dimB: dimension of subsystem B
    Returns:
        tuple: reduced density matrices for subsystems A and B
    """
    # explicit subsystem dimensions
    rho = np.reshape(rho, (dimA, dimB, dimA, dimB))
    # trace out subsystem B
    rhoA = np.trace(rho, axis1=1, axis2=3)
    # trace out subsystem A
    rhoB = np.trace(rho, axis1=0, axis2=2)
    return rhoA, rhoB
