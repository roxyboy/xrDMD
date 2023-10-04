import numpy as np
import pandas as pd
import xarray as xr
import dask.array as dsar

import scipy.signal as sps
import scipy.linalg as spl

import pytest
import numpy.testing as npt
import xarray.testing as xrt

import xrft
from xrDMD.DMDbase import (
    reconstruct,
)


@pytest.fixture
def sample_da_2d():
    x = np.linspace(-5, 5, 128)
    t = np.linspace(0, 4 * np.pi, 256)
    f1 = xr.DataArray(
        1.0 / np.cosh(x[np.newaxis, :] + 3) * np.exp(2.3j * t[:, np.newaxis]),
        coords=[("time", t), ("x", x)],
    )
    f2 = xr.DataArray(
        2.0 / np.cosh(x[np.newaxis, :]) * np.tanh(x) * np.exp(2.8j * t[:, np.newaxis]),
        coords=[("time", t), ("x", x)],
    )
    return f1, f2


def test_spectral_amplitude(sample_da_2d):
    X = sample_da_2d[0] + sample_da_2d[1]

    X_recon = reconstruct(
        X,
        dim="time",
        rank=5,
    )

    npt.assert_array_almost_equal(X, X_recon)
