# -*- coding: utf-8 -*-
"""Difference between star and all-to-all relative errors vs theory."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import compute_C_opt

### SIMULATION PARAMETERS ###
n_theory_min = 4
n_theory_max = 50
font_size = 18
figsize = (7, 5)

plt.rcParams["font.size"] = font_size

C_opt = compute_C_opt(n_theory_min, n_theory_max)


def _load_results(filename):
    return torch.load(os.path.join("data", filename))


C_2AtA = _load_results("C_AtA2_4to50")
C_3AtA = _load_results("C_AtA3_4to50")
C_4AtA = _load_results("C_AtA4_5to50")
C_5AtA = _load_results("C_AtA5_5to50")
C_2star = _load_results("C_Star2_4to50")
C_3star = _load_results("C_Star3_4to50")
C_4star = _load_results("C_Star4_5to50")
C_5star = _load_results("C_Star5_5to50")

teor_4to50 = torch.tensor(
    [C_opt[n] for n in range(4, 51)], dtype=torch.float64
)
teor_5to50 = torch.tensor(
    [C_opt[n] for n in range(5, 51)], dtype=torch.float64
)


def get_values(d):
    return torch.tensor(list(d.values()), dtype=torch.float64)


err2_A = torch.abs(teor_4to50 - get_values(C_2AtA)) / teor_4to50
err3_A = torch.abs(teor_4to50 - get_values(C_3AtA)) / teor_4to50
err4_A = torch.abs(teor_5to50 - get_values(C_4AtA)) / teor_5to50
err5_A = torch.abs(teor_5to50 - get_values(C_5AtA)) / teor_5to50
err2_S = torch.abs(teor_4to50 - get_values(C_2star)) / teor_4to50
err3_S = torch.abs(teor_4to50 - get_values(C_3star)) / teor_4to50
err4_S = torch.abs(teor_5to50 - get_values(C_4star)) / teor_5to50
err5_S = torch.abs(teor_5to50 - get_values(C_5star)) / teor_5to50

err2 = err2_S - err2_A
err3 = err3_S - err3_A
err4 = err4_S - err4_A
err5 = err5_S - err5_A

fig2, ax2 = plt.subplots(figsize=figsize)
ax2.grid(True, linestyle="--", alpha=0.7)

n_axis_4to50 = np.arange(4, 51)
n_axis_5to50 = np.arange(5, 51)

# ax2.scatter(n_axis_4to50, err2, color='b', label='2', s=25)
ax2.scatter(n_axis_4to50, err3, color="y", label="3", s=25)
ax2.scatter(n_axis_5to50, err4, color="m", label="4", s=25)
ax2.scatter(n_axis_5to50, err5, color="c", label="5", s=25)

ax2.set_xlim(3, 51)
ax2.set_xticks([4, 10, 20, 30, 40, 50])
ax2.set_xlabel("$n$")
ax2.set_ylabel(r"$\varepsilon (\boldsymbol{H}_{n,k}^{Star}) -\varepsilon (\boldsymbol{H}_{n,k}^{AtA})$")
ax2.legend(
    title="k",
    labelspacing=0.1,
    handletextpad=0.2,
)

plt.tight_layout()
plt.show()
