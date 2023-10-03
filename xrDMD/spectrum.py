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
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

from .dmd import Amatrix as _A

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


def _diff_coord(coord):
    """Returns the difference as a xarray.DataArray."""

    v0 = coord.values[0]
    calendar = getattr(v0, "calendar", None)
    if calendar:
        import cftime

        ref_units = "seconds since 1800-01-01 00:00:00"
        decoded_time = cftime.date2num(coord, ref_units, calendar)
        coord = xr.DataArray(decoded_time, dims=coord.dims, coords=coord.coords)
        return np.diff(coord)
    elif pd.api.types.is_datetime64_dtype(v0):
        return np.diff(coord).astype("timedelta64[s]").astype("f8")
    else:
        return np.diff(coord)


def _is_valid_dmd_coord(coord):
    return (
        is_numeric_dtype(coord)
        or is_datetime64_any_dtype(coord)
        or bool(getattr(coord[0].item(), "calendar", False))
    )


def _check_valid_dmd_coords(da, dim):
    if not np.all([_is_valid_dmd_coord(da.coords[d]) for d in dim]):
        raise ValueError(
            "All transformed dimensions coordinates must be numerical or datetime."
        )


def _get_coordinate_spacing(coord, spacing_tol):
    diff = _diff_coord(coord)
    delta = np.abs(diff[0])
    if not np.allclose(diff, diff[0], rtol=spacing_tol):
        raise ValueError(
            "Can't take Fourier transform because "
            "coodinate %s is not evenly spaced" % coord.name
        )
    if delta == 0.0:
        raise ValueError(
            "Can't take Fourier transform because spacing in coordinate %s is zero"
            % coord.name
        )
    return delta


def power_spectrum(
    da,
    spacing_tol=1e-3,
    dim=None,
    method=None,
    delay=0.0,
    eta=1.0,
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
        if len(N) == 2:
            beta = N[1] / N[0]
            lambd = np.sqrt(
                2 * (beta + 1)
                + (8 * beta / ((beta + 1) + np.sqrt(beta**2 + 14 * beta + 1)))
            )
            tau = lambd * np.sqrt(N[0]) * eta
            r = int(np.ceil(tau))
        else:
            r = int(np.max(M))
    else:
        r = int(rank)

    S, V, Atilde = _A(X, r, method=method)
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

    return ps[::2]
