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
from xrDMD.spectrum import (
    power_spectrum,
)


@pytest.fixture
def sample_da_1d():
    time = np.linspace(0, 10, 1000)
    dt = np.diff(time)[0]
    da = xr.DataArray(
        15 * np.sin(3 * 2 * np.pi * time)
        + 10 * np.sin(5 * 2 * np.pi * time) * np.exp(-time / 20),
        dims="time",
        coords={"time": time},
    )
    return da


def test_spectral_amplitude(sample_da_1d):
    N = len(sample_da_1d)

    ps = power_spectrum(
        sample_da_1d,
        dim="time",
        delay=200,
        rank=100,
    )

    ps *= 0.5
    ps /= xrft.power_spectrum(sample_da_1d).freq_time.spacing
    ps = np.sqrt(ps)
    ps = ps * (2.0 / N / np.diff(sample_da_1d.time)[0])

    npt.assert_allclose(np.array([15, 10]), np.sort(ps)[::-1][:2], atol=0.5)
    npt.assert_array_almost_equal(
        np.array([5, 3]), np.abs(ps.freq_time.imag)[ps.values.argsort()[-2:]]
    )

    ps = power_spectrum(
        sample_da_1d,
        dim="time",
        delay=200,
        rank=-1,
    )

    ps *= 0.5
    ps /= xrft.power_spectrum(sample_da_1d).freq_time.spacing
    ps = np.sqrt(ps)
    ps = ps * (2.0 / N / np.diff(sample_da_1d.time)[0])

    npt.assert_allclose(np.array([15, 10]), np.sort(ps)[::-1][:2], atol=0.5)
    npt.assert_array_almost_equal(
        np.array([5, 3]), np.abs(ps.freq_time.imag)[ps.values.argsort()[-2:]]
    )
