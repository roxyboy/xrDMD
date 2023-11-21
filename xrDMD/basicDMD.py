"""
Basis functions for dynamic-mode decomposing xarray data.
"""

import numpy as np
import xarray as xr
import scipy.linalg as spl

from .valid_coords import check_valid_dmd_coords as _check_valid_dmd_coords
from .valid_coords import get_coordinate_spacing as _get_coordinate_spacing

__all__ = [
    "svht",
    "Amatrix",
    "modes",
]


def svht(X, sv=None):
    # Singular Value Hard Threshold for sigma unknown
    m, n = sorted(X.shape)  # ensures m <= n
    beta = m / n  # ratio between 0 and 1
    if sv is None:
        sv = spl.svd(X, compute_uv=False)
    sv = np.squeeze(sv)
    omega_approx = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    tau = np.median(sv) * omega_approx

    return int(np.sum(sv > tau))


def Amatrix(da, dim, rank=None, method=None, compute_u=False):
    """
    Perform SVD decomposition to obtain A.

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
    compute_u : bool, optional
        Whether to compute also U in addition to S and V. Default is False.

    Returns
    -------
    S : ndarray
        The singular values.
    V : ndarray
        Unitary matrix having right singular vectors as columns.
    Atilde: ndarray
        Low dimensional linear model with `r \times r` dimensions.
    """

    X = da.isel({dim[0]: slice(None, -1)})
    Xp = da.isel({dim[0]: slice(1, None)})
    U, S, Vh = spl.svd(X.data, full_matrices=False)

    if rank == None:
        r = svht(X)
    else:
        r = int(rank)

    V = np.conj(Vh[:r].T)  # Hermitian transpose
    Uh = np.conj(U[..., :r].T)  # Hermitian transpose
    S = S[:r]

    Atilde = Uh @ Xp.data @ V @ spl.inv(np.diag(S))

    if method is not None:
        if method == "fb":
            Ub, Sb, Vbh = spl.svd(Xp.data, full_matrices=False)
            Vb = np.conj(Vbh[:r].T)  # Hermitian transpose
            Ubh = np.conj(Ub[..., :r].T)  # Hermitian transpose
            Sb = Sb[:r]

            b_Atilde = Ubh @ X.data @ Vb @ spl.inv(np.diag(Sb))

            Atilde = spl.sqrtm(Atilde @ spl.inv(b_Atilde))
        else:
            raise NotImplementedError(
                "Only forward-backward method is implemented for now."
            )

    if compute_u:
        return U, S, V, Atilde
    else:
        return S, V, Atilde


def modes(da, dim=None, spacing_tol=1e-3, rank=None, method=None):
    """
    Compute the DMD modes.

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
    method : str, optional
        Method of DMD.

    Returns
    -------
    Phi : xarray.DataArray
        DMD modes.
    omega : xarray.DataArray
        DMD frequencies.
    b: ndarray
        DMD amplitudes.
    """
    all_dim = list(da.dims)

    if dim is None:
        dim = all_dim
    else:
        if isinstance(dim, str):
            dim = [dim]

    if len(dim) > 1:
        raise ValueError("`dim` should be one-dimensional.")

    _check_valid_dmd_coords(da, dim)

    axis_num = [da.get_axis_num(d) for d in dim]
    not_dim = [all_dim[n] for n in range(da.ndim) if n not in axis_num]

    M = [da.shape[n] for n in axis_num]
    delta_t = [_get_coordinate_spacing(da[d], spacing_tol) for d in dim]

    da_stacked = da.stack(zeta=not_dim).transpose()
    X = da_stacked.isel({dim[0]: slice(None, -1)})
    Xp = da_stacked.isel({dim[0]: slice(1, None)})

    if rank == None:
        r = svht(X)
        if r == 0:
            r = min(X.shape)
    else:
        r = rank
    S, V, Atilde = Amatrix(da_stacked, dim, rank=r, method=method)

    lamb, W = spl.eig(Atilde)

    Phi = Xp.data @ V @ spl.inv(np.diag(S)) @ W  # DMD modes

    omega = np.log(lamb) / delta_t  # DMD frequencies

    x0 = X.isel({dim[0]: 0}).data
    b = spl.pinv(Phi) @ x0  # DMD amplitudes

    new_dims = ["mode"] + not_dim
    new_coords = dict()
    for d in new_dims:
        if d == "mode":
            new_coords[d] = np.arange(len(b))
        else:
            new_coords[d] = da.coords[d].data

    Phi = xr.DataArray(
        xr.DataArray(
            Phi,
            dims=["zeta", "mode"],
            coords={
                "zeta": da_stacked.zeta,
            },
        )
        .unstack("zeta")
        .data,
        dims=new_dims,
        coords=new_coords,
    )

    return (
        Phi,
        xr.DataArray(omega, coords=[("mode", new_coords["mode"])]),
        xr.DataArray(b, coords=[("mode", new_coords["mode"])]),
    )
