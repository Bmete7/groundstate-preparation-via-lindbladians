import os
import sys
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pickle
import pennylane as qml


def setup_paths():
    # Append project paths to sys.path
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    sim_dir = os.path.join(base_dir, "lindbladian_sim")
    sys.path.insert(0, base_dir)
    sys.path.insert(0, sim_dir)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, project_root)
    print("Project root added to sys.path:", project_root)
    print("Working directory:", os.getcwd())


setup_paths()
import lindbladian_sim as lbs


def generate_random_hamiltonian(d, rng):
    H = lbs.crandn((d, d), rng)
    H = 0.5 * (H + H.conj().T)
    eigvals, eigvec = np.linalg.eigh(H)
    return H, eigvals, eigvec


def plot_spectrum(eigvals):
    plt.plot(eigvals, ".")
    plt.xlabel(r"$i$")
    plt.ylabel(r"$\lambda_i$")
    plt.title("Eigenvalues of Hamiltonian")
    plt.show()


def plot_filter_function(fhat, Sw, da, b, db):
    wlist = np.linspace(-10, 10, 1001)
    plt.plot(wlist, fhat(wlist))
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\hat{f}(\omega)$")
    plt.title(
        f"Filter function (a = {Sw:.2f}, da = {da:.2f}, b = {b:.2f}, db = {db:.2f})"
    )
    plt.axvline(0, color="k", linestyle="--")
    plt.axvline(-Sw, color="r", linestyle=":")
    plt.show()


def partial_trace(rho, dimA, dimB):
    rho = rho.reshape((dimA, dimB, dimA, dimB))
    rhoA = np.trace(rho, axis1=1, axis2=3)
    rhoB = np.trace(rho, axis1=0, axis2=2)
    return rhoA, rhoB


def construct_dilated_operator(K, tau):
    zero_block = np.zeros_like(K)
    dilated_K = np.block([[zero_block, K.conj().T], [K, zero_block]])
    return expm(-1j * np.sqrt(tau) * dilated_K), dilated_K


def simulate_dilated_evolution(H, dilated_K, rho_init, tau, nsteps, label="unitary"):
    d = H.shape[0]
    ancilla = np.array([[1], [0]], dtype=complex) @ np.array([[1, 0]], dtype=complex)
    rho_principle = rho_init.copy()
    evolution_H = expm(-1j * tau * H)
    energies = [np.trace(H @ rho_principle).real]

    for _ in range(nsteps):
        rho = np.kron(ancilla, rho_principle)
        evolved = expm(-1j * np.sqrt(tau) * dilated_K)
        rho = evolved @ rho @ evolved.conj().T
        _, rho_principle = partial_trace(rho, 2, d)
        rho_principle = evolution_H @ rho_principle @ evolution_H.conj().T
        energies.append(np.trace(H @ rho_principle).real)

    return energies


def simulate_lindblad_superoperator(H, K, rho_init, tau, nsteps, with_coherent=True):
    if with_coherent:
        op = expm(
            tau * (lbs.hamiltonian_superoperator(H) + lbs.lindblad_operator_matrix(K))
        )
    else:
        op = expm(tau * lbs.lindblad_operator_matrix(K))

    d = H.shape[0]
    rho = rho_init.copy()
    energies = [np.trace(H @ rho).real]
    for _ in range(nsteps):
        rho = np.reshape(op @ rho.reshape(-1), (d, d))
        energies.append(np.trace(H @ rho).real)

    return energies


def main(save_unitaries=False):
    setup_paths()

    rng = np.random.default_rng(21)
    d = 16
    tau = 0.001
    nsteps = 10000

    H, eigvals, eigvec = generate_random_hamiltonian(d, rng)
    plot_spectrum(eigvals)

    gap = eigvals[1] - eigvals[0]
    Sw = np.linalg.norm(H, ord=2)
    a, da, b, db = 2.5 * Sw, 0.5 * Sw, gap, gap
    fhat = lambda w: lbs.filter_function(w, a, da, b, db)
    plot_filter_function(fhat, Sw, da, b, db)

    A = lbs.crandn((d, d), rng)
    K = lbs.construct_shoveling_lindblad_operator(A, H, fhat)

    rho_init = lbs.random_density_matrix(d, rng)
    expHK, expK = [
        expm(tau * m)
        for m in [
            lbs.hamiltonian_superoperator(H) + lbs.lindblad_operator_matrix(K),
            lbs.lindblad_operator_matrix(K),
        ]
    ]
    dilated_expK, dilated_K = construct_dilated_operator(K, tau)
    if save_unitaries:
        with open("data/evolutionH.pickle", "wb") as f:
            pickle.dump(expm(-1j * H * tau), f)

        with open("data/dilated_expK.pickle", "wb") as f:
            pickle.dump(dilated_expK, f)

    en_list = simulate_lindblad_superoperator(H, K, rho_init, tau, nsteps, True)
    # `en_list_bare` is a list that stores the energy expectation values of the system at each time
    # step during the simulation. It represents the evolution of the system without considering the
    # coherent part of the Lindblad superoperator. In other words, `en_list_bare` tracks the energy
    # evolution based solely on the Lindblad dissipative dynamics, without the additional coherent
    # evolution that is present in the full Lindblad superoperator.
    en_list_bare = simulate_lindblad_superoperator(H, K, rho_init, tau, nsteps, False)
    # The `en_list_unitary` list is storing the energy expectation values of the system at each time
    # step during the simulation. This particular list represents the evolution of the system with a
    # dilated unitary transformation. The function `simulate_dilated_evolution` is used to calculate
    # these energy values by evolving the system using the dilated operator constructed from the
    # Lindblad operator `K` and the Hamiltonian `H`. This approach includes both the coherent
    # evolution and the dissipative dynamics of the system.
    en_list_unitary = simulate_dilated_evolution(
        H, dilated_K, rho_init, tau, nsteps, "unitary"
    )
    print(dilated_K)
    # Plot all energy expectation value trajectories
    plt.semilogx(tau * np.arange(nsteps + 1), en_list, ".", label="With coherent part")
    plt.semilogx(tau * np.arange(nsteps + 1), en_list_bare, ".", label="Lindblad only")
    plt.semilogx(
        tau * np.arange(nsteps + 1), en_list_unitary, ".", label="Dilated unitary"
    )

    plt.semilogx(
        tau * np.arange(nsteps + 1),
        eigvals[0] * np.ones(nsteps + 1),
        label=r"$\lambda_0$",
    )
    plt.xlabel("Time")
    plt.ylabel(r"$\langle H \rangle$")
    plt.title(f"Convergence to ground state (d = {d})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
