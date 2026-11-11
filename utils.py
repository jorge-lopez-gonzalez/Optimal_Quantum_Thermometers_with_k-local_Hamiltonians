"""Shared helpers for plotting scripts (theoretical optimal thermometer precision)."""

import numpy as np
import scipy as sp


def compute_C_opt(n_min=4, n_max=50):
    """
  Compute the theoretical optimal C_opt for each system size n in [n_min, n_max].

  Returns a dict mapping n -> C_opt(n).
  """
    C_opt = {}
    for n in range(n_min, n_max + 1):
        def equation(x_val, n=n):
            return np.exp(x_val) - ((2**n - 1) * (x_val + 2)) / (x_val - 2)
        
        x_0 = 2.5
        x_sol = sp.optimize.fsolve(equation, x_0)[0]
        term_n = 2**n - 1
        C_opt[n] = (x_sol**2 * np.exp(x_sol) * term_n) / ((term_n + np.exp(x_sol)) ** 2)
    return C_opt
