import warnings
import operator
import sys
import functools as ft
from functools import reduce

import numpy as np
import xarray as xr
import pandas as pd

import dask.array as dsar

import scipy.linalg as spl

from .basicDMD import svht
from .valid_coords import check_valid_dmd_coords as _check_valid_dmd_coords
from .valid_coords import get_coordinate_spacing as _get_coordinate_spacing

__all__ = ["power_spectrum"]


def _freq(N, delta_x):
    # calculate frequencies from coordinates
    # coordinates are always loaded eagerly, so we use numpy
    fftfreq = [np.fft.fftfreq] * len(N)

    k = [fftfreq(Nx, dx) for (fftfreq, Nx, dx) in zip(fftfreq, N, delta_x)]

    return k


def _new_dims_and_coords(da, dim, wavenm, prefix):
    # set up new dimensions and coordinates for dataarray
    swap_dims = dict()
    new_coords = dict()
    wavenm = dict(zip(dim, wavenm))

    for d in dim:
        k = wavenm[d]
        new_name = prefix + d if d[: len(prefix)] != prefix else d[len(prefix) :]
        new_dim = xr.DataArray(k, dims=new_name, coords={new_name: k}, name=new_name)
        new_dim.attrs.update({"spacing": k[1] - k[0]})
        new_coords[new_name] = new_dim
        swap_dims[d] = new_name

    return new_coords, swap_dims


def _Amatrix(X, rank, method=None):
    """
    A matrix specific to time-delayed coordinates.

    Parameters
    ----------
    X : ndarray
        The data to DMD.
    rank : int
        Rank to truncate SVD.
    method : str
        Method of DMD.

    Returns
    -------
    S : ndarray
        The singular values.
    V : ndarray
        Unitary matrix having right singular vectors as columns.
    Atilde: ndarray
        Low dimensional linear model with `r \times r` dimensions.
    """
    r = rank

    U, S, Vh = spl.svd(X[..., :-1].data, full_matrices=False)
    V = np.conj(Vh[:r].T)  # Hermitian transpose
    Uh = np.conj(U[..., :r].T)  # Hermitian tranpose

    Atilde = Uh @ X[..., 1:] @ V @ spl.inv(np.diag(S[:r]))

    if method is not None:
        if method == "fb":
            Ub, Sb, Vbh = spl.svd(X[..., 1:], full_matrices=False)
            Vb = np.conj(Vbh[:r].T)  # Hermitian transpose
            Ubh = np.conj(Ub[..., :r].T)  # Hermitian transpose

            b_Atilde = Ubh @ X[..., :-1] @ Vb @ spl.inv(np.diag(Sb[:r]))

            Atilde = spl.sqrtm(Atilde @ spl.inv(b_Atilde))
        else:
            raise NotImplementedError(
                "Only forward-backward method is implemented for now."
            )

    return S[:r], V, Atilde


def power_spectrum(
    da,
    spacing_tol=1e-3,
    dim=None,
    method=None,
    delay=0.0,
    eta=0.0,
    rank=None,
    scaling="density",
    prefix="freq_",
):
    """
    Compute power spectrum based on DMD.

    Parameters
    ----------
    da : xarray.DataArray
        The data to take the power spectrum.
    spacing_tol: float, optional
        Spacing tolerance. Frequency spectrum should not be applied to uneven grid but
        this restriction can be relaxed with this setting. Use caution.
    dim : str or sequence of str, optional
        The dimensions along which to take the transformation. If `None`, all
        dimensions will be transformed. If the inputs are dask arrays, the
        arrays must not be chunked along these dimensions.
    method : str
        Method of DMD.
    delay : int or float
        Number of indices to construct time-delayed coordinates.
    eta : float
        A priori uncertainty in the data. It is ignored when `rank` is prescribed.
    rank : int
        Rank to truncate SVD.
    scaling : str, optional
        If 'density', it will normalize the output to power spectral density.
        If 'spectrum', it will normalize the output to power spectrum.

    Returns
    -------
    ps : xarray.DataArray
        Power spectrum.
    """
    if dim is None:
        dim = list(da.dims)
    else:
        if isinstance(dim, str):
            dim = [dim]

    if len(dim) > 1:
        raise NotImplementedError(
            "Spectrum for multidimensional data is not implemented yet."
        )

    _check_valid_dmd_coords(da, dim)

    axis_num = [da.get_axis_num(d) for d in dim]
    M = [da.shape[n] for n in axis_num]
    delta_t = [_get_coordinate_spacing(da[d], spacing_tol) for d in dim]

    if delay > 0.0:  # time-delayed coordinates
        delay = int(delay)
        for i in range(delay):
            if i == 0:
                X = np.array(
                    [
                        da.isel({dim[0]: slice(None, -delay - 1)}).data,
                        da.isel({dim[0]: slice(i + 1, -delay)}).data,
                    ]
                )
            else:
                X = np.concatenate(
                    (X, np.array([da.isel({dim[0]: slice(i + 1, -delay + i)}).data])),
                    axis=0,
                )
        X = np.concatenate(
            (X, np.array([da.isel({dim[0]: slice(delay + 1, None)}).data])), axis=0
        )
    else:
        X = da

    N = X.shape

    if rank is None:
        if eta != 0.0:
            m, n = sorted(X.shape)  # ensures m <= n
            beta = m / n  # ratio between 0 and 1
            lambd = np.sqrt(
                2 * (beta + 1)
                + (8 * beta / ((beta + 1) + np.sqrt(beta**2 + 14 * beta + 1)))
            )
            tau = lambd * np.sqrt(n) * eta
            r = int(np.ceil(tau))
        else:
            r = svht(X)
    else:
        r = int(rank)

    S, V, Atilde = _Amatrix(X, r, method=method)
    lamb, W = spl.eig(Atilde)

    fbDMDfreqs = [(np.log(lamb) / (2 * np.pi * np.prod(delta_t)))]

    Phi = X[..., 1:] @ V @ spl.inv(np.diag(S)) @ W
    b = spl.pinv(Phi) @ X[..., 0]

    fbDMDpower = (np.abs(b) / np.sqrt(delay) * (np.prod(delta_t) * np.prod(M))) ** 2

    ftfreqs = _freq(M, delta_t)
    ftfreqs = dict(zip(dim, ftfreqs))
    tmp_coords = dict()
    for d in dim:
        f = ftfreqs[d]
        tmp_name = prefix + d if d[: len(prefix)] != prefix else d[len(prefix) :]
        tmp_dim = xr.DataArray(f, dims=tmp_name, coords={tmp_name: f}, name=tmp_name)
        tmp_dim.attrs.update({"spacing": f[1] - f[0]})
        tmp_coords[tmp_name] = tmp_dim

    fs = np.prod([float(tmp_coords[prefix + d].spacing) for d in dim])
    if scaling == "density":
        ps = fbDMDpower * np.abs(fs)
    elif scaling == "spectrum":
        ps = fbDMDpower * np.abs(fs) ** 2
    else:
        raise ValueError("Unrecognized scaling convention.")

    newcoords, swap_dims = _new_dims_and_coords(ps, dim, fbDMDfreqs, prefix)
    ps = xr.DataArray(
        ps, dims=da.dims, coords=dict([c for c in da.coords.items() if c[0] not in dim])
    )
    ps = ps.swap_dims(swap_dims).assign_coords(newcoords)
    ps = ps.drop([d for d in dim if d in ps.coords])

    return ps[::2] * 2
