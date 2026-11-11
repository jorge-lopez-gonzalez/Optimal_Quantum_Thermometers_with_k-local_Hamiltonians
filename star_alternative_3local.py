# -*- coding: utf-8 -*-
"""Star 3-local Hamiltonian (alternative geometry)."""

import random
import time

import torch

### SIMULATION PARAMETERS ###
plot_spectrum = False # True to plot the resulting spectrum, false if not
n_min = 4
n_max = 5 # Excluded
num_iterations = 10000
learning_rate = 0.001
progress_interval = 100
plot_spectrum = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma_0 = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32, device=device)
sigma_3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32, device=device)
sigma_dict = {0: sigma_0, 3: sigma_3}


def tensor_product(indices):
    matrices = [sigma_dict[k] for k in indices]
    result = matrices[0]
    for mat in matrices[1:]:
        result = torch.kron(result, mat)
    return result


C_max = {}
alpha_max = {}


def build_star_alternative_basis(n):
    tensor_products = {}
    alpha_keys = []

    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0

    for pos in range(n):
        indices = [0] * n
        indices[pos] = 3
        tensor_products[tuple(indices)] = tensor_product(indices)
        alpha_keys.append((pos,))
        count_1 += 1

    for pos in range(1, n):
        indices = [0] * n
        indices[0] = 3
        indices[pos] = 3
        tensor_products[tuple(indices)] = tensor_product(indices)
        alpha_keys.append((0, pos))
        count_2 += 1

    for pos in range(1, n):
        indices = [0] * n
        indices[pos] = 3
        pos2 = 1 if pos == n - 1 else pos + 1
        indices[pos2] = 3
        tensor_products[tuple(indices)] = tensor_product(indices)
        alpha_keys.append((pos, pos2))
        count_3 += 1

    for pos in range(1, n):
        indices = [0] * n
        indices[0] = 3
        indices[pos] = 3
        pos3 = 1 if pos == n - 1 else pos + 1
        indices[pos3] = 3
        tensor_products[tuple(indices)] = tensor_product(indices)
        alpha_keys.append((0, pos, pos3))
        count_4 += 1

    return tensor_products, alpha_keys, count_1, count_2, count_3, count_4


def objective_function(alpha_tensor, n, tensor_products, count_1, count_2, count_3, count_4):
    H = torch.zeros(2**n, 2**n, dtype=torch.float32, device=device)
    n_terms = count_1 + count_2 + count_3 + count_4
    terms = list(tensor_products.values())

    for i in range(n_terms):
        if i < count_1:
            if i == 0:
                H = H + alpha_tensor[0] * terms[0]
            else:
                H = H + alpha_tensor[1] * terms[i]
        elif i < count_1 + count_2:
            H = H + alpha_tensor[2] * terms[i]
        elif i < count_1 + count_2 + count_3:
            H = H + alpha_tensor[3] * terms[i]
        else:
            H = H + alpha_tensor[4] * terms[i]

    eigenvalues = torch.diagonal(H)

    z = torch.sum(torch.exp(-eigenvalues))
    p = torch.exp(-eigenvalues) / z

    esp_H = torch.sum(p * eigenvalues)
    esp_H_cuad = torch.sum(p * eigenvalues**2)

    if plot_spectrum:
        return esp_H_cuad - esp_H**2, eigenvalues
    else:
        return esp_H_cuad - esp_H**2


for n in range(n_min, n_max):
    t_start = time.time()
    print(f"Number of spins: {n}")

    tensor_products, alpha_keys, c1, c2, c3, c4 = build_star_alternative_basis(n)

    alpha_tensor = torch.tensor(
        [random.random() for _ in range(5)],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.Adam([alpha_tensor], lr=learning_rate, maximize=True)

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        if plot_spectrum:
            fun, eigenvalues = objective_function(
                alpha_tensor, n, tensor_products, c1, c2, c3, c4
            )
        else: 
            fun = objective_function(
                alpha_tensor, n, tensor_products, c1, c2, c3, c4
            )
        fun.backward()
        optimizer.step()

        if (iteration + 1) % progress_interval == 0:
            print(f"Iteration {iteration + 1}: objective = {fun.item()}")

    print(f"Optimal coefficients alpha = {alpha_tensor}")
    t_end = time.time()
    print(f"Runtime: {t_end - t_start:.2f} seconds")

    C_max[n] = fun.item()
    alpha_max[n] = alpha_tensor

    if plot_spectrum:
        import matplotlib.pyplot as plt

        spectrum = eigenvalues.detach().cpu().tolist()
        spectrum.sort()
        plt.scatter(range(2**n), spectrum)
        plt.ylabel("Energy eigenvalue")
        plt.xlabel("Eigenstate index")
        plt.title(f"Eigenvalue spectrum for n={n}")
        plt.tight_layout()
        plt.show()
