from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

# start with computation helpers for EB

"""
EB (equation 5) from the paper:
    EB = C_S * sum_k h_k * eps(t_k)  +  C_L * sum_k h_k^2 * L(t_k)

where: 
h_k = t_k - t_{k-1} = step size
eps(t) = flow matching error at t = (FME, analogue of score matching loss)
L(t) = smoothness term

Note:
for RF with straight paths, scoere smoothness is roughly constant. 
the velocity field is trained to be linear, we default to L(t) = 1.
Lipschtiz setting - can also pass L(t) = 1/sigm_t^4 (early stopping)

E(t, s) shorthand (from section 4 of paper):
    E(t, s) = C_S * eps(t) * (s - t)  +  C_L * L(t) * (s - t)^2

""" 

def compute_E(
    t: float,
    s: float, 
    eps_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    C: float = 1.0,
) -> float:

    """
    compute E(t, s) = eps(t)*(s-t) + C * L(t)*(s-t)^2
    C is C_L / C_S : ratio hyperparameter. We are working in unts where C_S = 1.
    """

    h = s - t
    return eps_fn(t) * h + C * L_fn(t) * h * h

def compute_EB(
    schedule: List[float],
    eps_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    C: float = 1.0,
) -> float: 
    
    """
    COmpute the total EB for a given schedule. 
    Schedule: [t_0 = 0, t_1, .., t_N = 1]
    """

    total = 0.0
    for k in range(1, len(schedule)):
        total += compute_E(schedule[k - 1], schedule[k], eps_fn, L_fn, C)
    return total

# default smoothness functions

def L_lipschitz(_t: float) -> float:
    """ LIpschitze setting: L(t) = 1 (constant)."""
    return 1.0

def L_early_stopping(t: float, eps: float = 1e-3) -> float:
    
    """
    early stopping setting: L(t) = 1 / sigma_t^4.
    for RF: sigma_t ~ sqrt(t*(1-t)), so we use sigma_t = sqrt(t).
    We clamp t away from 0, avoiding division by zero.
    """

    t_safe = max(t, eps)
    # variance will grow linearly with t in RF
    sigma_sq = t_safe
    return 1.0 / (sigma_sq ** 2 + eps)

# GA implementation (Algorithm 1)

# GC implementation (algorithm 2)

# schedule builder / utilities.