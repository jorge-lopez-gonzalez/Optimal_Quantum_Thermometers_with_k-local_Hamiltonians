# -*- coding: utf-8 -*-
"""General 3-local Pauli Hamiltonian optimization (full Pauli or Z-only basis)."""

import itertools
import random
import time

import torch

### SIMULATION PARAMETERS ###
use_full_pauli = False  # True: X, Y, Z Pauli strings + eigvalsh; False: Z only + diagonal(H)
plot_spectrum = False # True to plot the resulting spectrum, false if not
n_min = 4
n_max = 5 # Excluded
num_iterations = 10000
learning_rate = 0.001
progress_interval = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if use_full_pauli:
    sigma_0 = torch.tensor([[1, 0], [0, 1]], dtype=torch.cfloat, device=device)
    sigma_1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=device)
    sigma_2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=device)
    sigma_3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)
    sigma_dict = {0: sigma_0, 1: sigma_1, 2: sigma_2, 3: sigma_3}
    ham_dtype = torch.cfloat
else:
    sigma_0 = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32, device=device)
    sigma_3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32, device=device)
    sigma_dict = {0: sigma_0, 3: sigma_3}
    ham_dtype = torch.float32


def tensor_product(indices):
    matrices = [sigma_dict[k] for k in indices]
    result = matrices[0]
    for mat in matrices[1:]:
        result = torch.kron(result, mat)
    return result


def build_tensor_basis(n):
    """Build k-local (up to 3-body) Pauli strings and coefficient keys."""
    tensor_products = {}
    alpha_keys = []

    if use_full_pauli:
        pauli_indices = range(1, 4)
        for i in pauli_indices:
            for pos in range(n):
                indices = [0] * n
                indices[pos] = i
                tensor_products[tuple(indices)] = tensor_product(indices)
                alpha_keys.append((pos, i))

        for i, j in itertools.product(pauli_indices, repeat=2):
            for pos1, pos2 in itertools.combinations(range(n), 2):
                indices = [0] * n
                indices[pos1] = i
                indices[pos2] = j
                tensor_products[tuple(indices)] = tensor_product(indices)
                alpha_keys.append((pos1, pos2, i, j))

        for i, j, k in itertools.product(pauli_indices, repeat=3):
            for pos1, pos2, pos3 in itertools.combinations(range(n), 3):
                indices = [0] * n
                indices[pos1] = i
                indices[pos2] = j
                indices[pos3] = k
                tensor_products[tuple(indices)] = tensor_product(indices)
                alpha_keys.append((pos1, pos2, pos3, i, j, k))
    else:
        for pos in range(n):
            indices = [0] * n
            indices[pos] = 3
            tensor_products[tuple(indices)] = tensor_product(indices)
            alpha_keys.append((pos,))

        for pos1, pos2 in itertools.combinations(range(n), 2):
            indices = [0] * n
            indices[pos1] = 3
            indices[pos2] = 3
            tensor_products[tuple(indices)] = tensor_product(indices)
            alpha_keys.append((pos1, pos2))

        for pos1, pos2, pos3 in itertools.combinations(range(n), 3):
            indices = [0] * n
            indices[pos1] = 3
            indices[pos2] = 3
            indices[pos3] = 3
            tensor_products[tuple(indices)] = tensor_product(indices)
            alpha_keys.append((pos1, pos2, pos3))

    return tensor_products, alpha_keys


def objective_function(alpha_tensor, n, tensor_products):
    H = torch.zeros(2**n, 2**n, dtype=ham_dtype, device=device)
    for i, term in enumerate(tensor_products.values()):
        H = H + alpha_tensor[i] * term

    if use_full_pauli:
        eigenvalues = torch.linalg.eigvalsh(H)
    else:
        eigenvalues = torch.diagonal(H)

    z = torch.sum(torch.exp(-eigenvalues))
    p = torch.exp(-eigenvalues) / z

    esp_H = torch.sum(p * eigenvalues)
    esp_H_cuad = torch.sum(p * eigenvalues**2)

    objective = esp_H_cuad - esp_H**2
    if plot_spectrum:
        return objective, eigenvalues
    return objective


C_max = {}
alpha_max = {}

for n in range(n_min, n_max):
    t_start = time.time()
    print(f"Number of spins: {n}")

    tensor_products, alpha_keys = build_tensor_basis(n)
    n_terms = len(tensor_products)

    alpha_tensor = torch.tensor(
        [random.uniform(-1, 1) for _ in range(n_terms)],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam([alpha_tensor], lr=learning_rate, maximize=True)

    last_eigenvalues = None
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        out = objective_function(alpha_tensor, n, tensor_products)
        if plot_spectrum:
            fun, last_eigenvalues = out
        else:
            fun = out

        fun.backward()
        optimizer.step()

        if (iteration + 1) % progress_interval == 0:
            print(f"Iteration {iteration + 1}: objective = {fun.item()}")

    print(f"Optimal coefficients alpha = {alpha_tensor}")
    t_end = time.time()
    print(f"Runtime: {t_end - t_start:.2f} seconds")

    C_max[n] = fun.item()
    alpha_dic = {alpha_keys[i]: alpha_tensor[i] for i in range(n_terms)}
    alpha_max[n] = alpha_dic

    if plot_spectrum and last_eigenvalues is not None:
        import matplotlib.pyplot as plt

        spectrum = last_eigenvalues.detach().cpu().tolist()
        spectrum.sort()
        plt.scatter(range(2**n), spectrum)
        plt.ylabel("Energy eigenvalue")
        plt.xlabel("Eigenstate index")
        plt.title(f"Eigenvalue spectrum for n={n}")
        plt.tight_layout()
        plt.show()
