# Optimal Quantum Thermometers with k-local Hamiltonians

PyTorch implementation accompanying the paper **"Optimal Quantum Thermometers with k-local Hamiltonians"**.

This repository provides code to optimize thermal precision in spin systems governed by **k-local interactions** under **All-to-all** and **Star** network topologies. It implements custom optimization loops using PyTorch (with GPU support) to maximize heat capacity, comparing numerical results directly against exact analytical bounds.

## Requirements

* Python 3.9+

* [PyTorch](https://pytorch.org/)

* NumPy, SciPy, Matplotlib

```
pip install torch numpy scipy matplotlib

```

GPU acceleration is optional and recommended for large system sizes: optimization scripts automatically select `cuda` when available, falling back to `cpu` otherwise.

## Repository layout

| Script | Description | 
| ----- | ----- | 
| `general_3local.py` | 3-local general Pauli-string models optimizations | 
| `all_to_all_3local.py` | All-to-all **3-local** optimizations | 
| `star_3local.py` | Star **3-local** optimizations | 
| `star_alternative_3local.py` | Alternative geometry Star 3-local optimizations (Appendix B) | 
| `all_to_all_klocal.py` | All-to-all **k-local** optimizations | 
| `star_klocal.py` | Star **k-local** optimizations | 
| `error_AtA.py` | Plot: Relative error for All-to-all numerical vs theoretical optima | 
| `error_stars.py` | Plot: Relative error for Star numerical vs theoretical optima | 
| `error_star_vs_AtA.py` | Plot: Difference between Star and All-to-all relative errors | 
| `utils.py` | Shared theoretical optima calculation via `compute_C_opt()` | 
| `data/` | Directory containing saved result dicts (`torch.load`) for plot generation | 

## Simulation Parameters & Optimization Guidelines

Each optimization script starts with a standardized **`### SIMULATION PARAMETERS ###`** block (`n_min`, `n_max`, `learning_rate`, `num_iterations`, `k_locality`, `estimation_scale`, `plot_spectrum`, etc.). Edit this block to configure individual runs.

> **Note on Default Parameters:** The repository is pre-configured with small system sizes ($n$) as fast smoke-test ranges. To reproduce the full paper results across large system sizes, increase `n_max` progressively.

### Optimization Guidelines & Practical Considerations (Appendix A)

Numerical optimization of $k$-local quantum models using the Adam algorithm in PyTorch presents non-trivial challenges as system size $n$ increases due to non-convex landscapes and local optima. Based on **Appendix A** of the manuscript, keep the following strategies in mind when configuring parameters:

1. **Progressive Optimization over** $n$**:** It is strongly recommended to run simulations in small incremental ranges of $n$, verifying the convergence. Convergence is faster for small systems, and their converged coefficient parameters ($\boldsymbol{\alpha}$) can serve as optimal initialization vectors for larger system sizes.

2. **Hyperparameter Adaptation:**

   * **Learning Rate & Iterations:** Increase `num_iterations` and lower `learning_rate` (or implement decay schedules) for larger $n$ to avoid overshooting local minima.

   * **Parameter Initialization:** Initialize coefficients randomly within controlled intervals or seed them directly from $n - 1$ spin optimal configurations.

3. **Objective Rescaling (`estimation_scale`):** For large systems or difficult landscape convergence, setting `estimation_scale` normalizes the objective heat capacity by an estimated target value (e.g., using $C_{\text{opt}}$ from $n - 1$ plus an expected incremental offset). This facilitates smooth gradient steps.

>  Limitations & Disclaimer
> - **Numerical Convergence & Local Optima:** Results may occasionally suffer from slow convergence or get trapped in local optima, especially as system size $n$ increases, not getting the real optima. 
> - **Research Code & Performance:** This repository contains research code developed for academic exploration and reproducing the paper's results. While functional and verified, the scripts are not fully optimized for maximal performance or memory efficiency and can be further streamlined for larger-scale applications.

## Key Script Variables & Flags

Before launching simulations, ensure the following flags are set appropriately:

### `general_3local.py`

* `use_full_pauli = True`: Evaluates 1–3 body Pauli $X, Y, Z$ strings. Requires full Hamiltonian matrix diagonalization via `torch.linalg.eigvalsh`.

* `use_full_pauli = False`: Restricts interactions strictly to $Z$ strings. The Hamiltonian is already diagonal, making computation significantly faster and more memory-efficient.

  *Note: For all models other than `general_3local.py` (with `use_full_pauli=True`), only* $Z$ *strings are considered, as justified in the manuscript.*

### `plot_spectrum` Flag

* `plot_spectrum = True`: Computes, tracks, and visualizes the full energy eigenvalue spectrum during training.

* `plot_spectrum = False`: Skips energy spectrum plotting and some calculations that may not be neccesary.

* **Memory Warning:** Setting `plot_spectrum = True` requires storing and calculating full eigenvalue arrays at every iteration. For large system sizes $n$, this induces high computational overhead and can lead to GPU/RAM memory exhaustion. It is recommended to leave `plot_spectrum = False` for large-scale runs.

## Running optimizations

Execute scripts directly from the repository root:

```
python general_3local.py          
python all_to_all_3local.py
python star_3local.py
python star_alternative_3local.py
python all_to_all_klocal.py
python star_klocal.py

```

Outputs are evaluated in memory as `C_max` and `alpha_max` and can be save into the `data/` folder for subsequent visualization using `torch.save()`.

## Error figures

The visualization scripts (`error_AtA.py`, `error_stars.py`, `error_star_vs_AtA.py`) generate publication-ready plots comparing numerical optima against analytical bounds:

> **Data Source Notice:** By default, the plotting scripts are configured to load the benchmark datasets obtained in the paper (stored  in `./data/`). However, these scripts can be adapted to plot custom simulation outputs saved from your own optimization runs into `./data/`.

### Reproducing Benchmark Plots

1. Verify that serialized result dictionaries exist in `data/` under standard filenames:

   * `C_AtA2_4to50`, `C_AtA3_4to50`, `C_AtA4_5to50`, `C_AtA5_5to50`

   * `C_Star2_4to50`, `C_Star3_4to50`, `C_Star4_5to50`, `C_Star5_5to50`

2. Run the plot commands:

```
python error_AtA.py
python error_stars.py
python error_star_vs_AtA.py

```

Generated plots match the formatting, legends, and styling present in the published paper.

## Citation

If you use this code, please cite the accompanying paper: http://arxiv.org/abs/2611.18327
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
