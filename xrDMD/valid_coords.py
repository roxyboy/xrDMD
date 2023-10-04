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


def check_valid_dmd_coords(da, dim):
    if not np.all([_is_valid_dmd_coord(da.coords[d]) for d in dim]):
        raise ValueError(
            "All transformed dimensions coordinates must be numerical or datetime."
        )


def get_coordinate_spacing(coord, spacing_tol):
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
