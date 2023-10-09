"""
Reconstruction for dynamic-mode decomposing xarray data.
"""

import numpy as np
import xarray as xr
import scipy.linalg as spl

from .valid_coords import check_valid_dmd_coords as _check_valid_dmd_coords
from .valid_coords import get_coordinate_spacing as _get_coordinate_spacing

__all__ = [
    "reconstruct",
]


def reconstruct(da, dim=None, spacing_tol=1e-3, rank=None, method=None, mode='basic'):
    """
    Reconstruct da using DMDs.

    Parameters
    ----------
    da : xarray.DataArray
        The data to DMD.
    dim : str
        Dimension over SVD is taken.
    rank : int
        Rank to truncate SVD.
    method : str
        Method of DMD.
    mode : str
        Flavor of DMD.

    Returns
    -------
    da_recon : xarray.DataArray
        DMD reconstruction of `da`.
    """
    if mode == 'basic':
        from .basicDMD import modes
    elif mode == 'mr':
        from .mrDMD import modes
    else:
        raise NotImplementedError(
        "Only basic and multi-resolution DMD are implemented."
        )
        
    Phi, omega, b = modes(da, dim=dim, spacing_tol=1e-3, rank=5, method=method)

    time_dynamics = b * np.exp(omega * da.time)

    da_recon = Phi.data.T @ time_dynamics.data

    da_recon = xr.DataArray(da_recon.T, dims=da.dims, coords=da.coords)

    return da_recon