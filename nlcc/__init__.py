"""
NLCC: Nonlinear Cross Correlation (Ding et al. 1997).
Import as a library: use nlcc.io for data loading, nlcc.nlcc for the algorithm.
"""

from nlcc.io import load_inp, load_signal, load_pair
from nlcc.nlcc import (
    range_normalize,
    zscore,
    embed,
    conditional_dispersion_curves,
    nlcc_metrics_from_curves,
    eps_at_level,
    eps_grid_from_inp_matlab,
    eps_grid_from_config,
)

__all__ = [
    "load_inp",
    "load_signal",
    "load_pair",
    "range_normalize",
    "zscore",
    "embed",
    "conditional_dispersion_curves",
    "nlcc_metrics_from_curves",
    "eps_at_level",
    "eps_grid_from_inp_matlab",
    "eps_grid_from_config",
]
