# -*- coding: utf-8 -*-
"""Analytic all-to-all k-local Hamiltonian:"""

import random
import time
from math import comb

import torch

### SIMULATION PARAMETERS ###
plot_spectrum = False # True to plot the resulting spectrum, false if not. Plotting the spectrum requires more memory and time.
n_min = 7
n_max = 8 # Excluded
k_locality = 4
num_iterations = 10000
learning_rate = 0.001
estimation_scale = 7 # It is useful, specially for large n, to give an estimation to normalize the objective_function, this estimation could typically be the result for n-1
progress_interval = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C_max = {}
alpha_max = {}


def objective_function(alpha_tensor, n, k_locality):
    k_values = torch.arange(n + 1, dtype=torch.float32, device=device)
    degeneracies = torch.tensor(
        [comb(n, int(k.item())) for k in k_values],
        dtype=torch.int64,
        device=device,
    )
    eigenvalues = torch.zeros_like(k_values)

    for i in range(1, k_locality + 1):
        coeff = alpha_tensor[i - 1]
        E_k_i = torch.zeros_like(k_values)
        for j in range(0, i + 1):
            signo = (-1) ** (i - j)
            bin_k_i = torch.tensor(
                [comb(int(k.item()), j) for k in k_values], device=device
            )
            bin_nk_i = torch.tensor(
                [comb(int(n - k.item()), i - j) for k in k_values], device=device
            )
            E_k_i = E_k_i + signo * bin_k_i * bin_nk_i
        eigenvalues = eigenvalues + coeff * E_k_i

    eigenvalues_shifted = eigenvalues - eigenvalues.min()

    z = torch.sum(degeneracies * torch.exp(-eigenvalues_shifted))
    p = torch.exp(-eigenvalues_shifted) / z

    esp_H = torch.sum(degeneracies * p * eigenvalues)
    esp_H_cuad = torch.sum(degeneracies * p * eigenvalues**2)

    if plot_spectrum:
        eigenvalues_full = eigenvalues.repeat_interleave(degeneracies)
        return (esp_H_cuad - esp_H**2) / estimation_scale, eigenvalues_full
    else:
        return (esp_H_cuad - esp_H**2) / estimation_scale


for n in range(n_min, n_max):
    t_start = time.time()
    print(f"Number of spins: {n}")

    alpha_tensor = torch.tensor(
        [random.uniform(-1, 1) for _ in range(k_locality)],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )
    if plot_spectrum:
        while torch.isnan(objective_function(alpha_tensor, n, k_locality)[0]) or (
            objective_function(alpha_tensor, n, k_locality)[0] == 0
        ):
            alpha_tensor = torch.tensor(
                [random.random() for _ in range(k_locality)],
                requires_grad=True,
                dtype=torch.float32,
                device=device,
            )
    else: 
        while torch.isnan(objective_function(alpha_tensor, n, k_locality)) or (
            objective_function(alpha_tensor, n, k_locality) == 0
        ):
            alpha_tensor = torch.tensor(
                [random.random() for _ in range(k_locality)],
                requires_grad=True,
                dtype=torch.float32,
                device=device,
            )

    optimizer = torch.optim.Adam([alpha_tensor], lr=learning_rate, maximize=True)

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        if plot_spectrum:
            fun, eigenvalues = objective_function(alpha_tensor, n, k_locality)
        else:
            fun = objective_function(alpha_tensor, n, k_locality)
        fun.backward()
        optimizer.step()

        if (iteration + 1) % progress_interval == 0:
            print(f"Iteration {iteration + 1}: objective = {estimation_scale * fun.item()}")

    print(f"Optimal coefficients alpha = {alpha_tensor}")
    t_end = time.time()
    print(f"Runtime: {t_end - t_start:.2f} seconds")

    C_max[n] = estimation_scale * fun.item()
    alpha_max[n] = alpha_tensor
    
    if plot_spectrum:
        import matplotlib.pyplot as plt

        spectrum = eigenvalues.detach().cpu().tolist()
        spectrum.sort()
        plt.scatter(range(2**n), spectrum)
        plt.ylabel("Energy eigenvalue")
        plt.xlabel("Eigenstate index")
        plt.tight_layout()
        plt.show()
