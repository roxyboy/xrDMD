import numpy as np
import pandas as pd
import xarray as xr

import scipy.signal as sps
import scipy.linalg as spl

import pytest
import numpy.testing as npt
import xarray.testing as xrt

from xrDMD.mrDMD import (
    mrdmd,
)


def _create_sample_data(x, t):
    Xm, Tm = np.meshgrid(x, t)

    D = np.exp(-np.power(Xm / 2, 2)) * np.exp(0.8j * Tm)
    D += np.sin(0.9 * Xm) * np.exp(1j * Tm)
    D += np.cos(1.1 * Xm) * np.exp(2j * Tm)
    D += 0.6 * np.sin(1.2 * Xm) * np.exp(3j * Tm)
    D += 0.6 * np.cos(1.3 * Xm) * np.exp(4j * Tm)
    D += 0.2 * np.sin(2.0 * Xm) * np.exp(6j * Tm)
    D += 0.2 * np.cos(2.1 * Xm) * np.exp(8j * Tm)
    D += 0.1 * np.sin(5.7 * Xm) * np.exp(10j * Tm)
    D += 0.1 * np.cos(5.9 * Xm) * np.exp(12j * Tm)
    D += 0.1 * np.random.randn(*Xm.shape)
    D += 0.03 * np.random.randn(*Xm.shape)
    D += 5 * np.exp(-np.power((Xm + 5) / 5, 2)) * np.exp(-np.power((Tm - 5) / 5, 2))
    D[:800, 40:] += 2
    D[200:600, 50:70] -= 3
    D[800:, :40] -= 2
    D[1000:1400, 10:30] += 3
    D[1000:1080, 50:70] += 2
    D[1160:1240, 50:70] += 2
    D[1320:1400, 50:70] += 2

    D = xr.DataArray(D, dims=["time", "x"], coords={"time": t, "x": x})
    return D


@pytest.fixture
def sample_data():
    x = np.linspace(-10, 10, 80)
    t = np.linspace(0, 20, 1600)
    return _create_sample_data(x, t)


def test_mrdmd(sample_data):
    reX, _, _, _ = mrdmd(
        sample_data,
        dim="time",
        rank=None,
        delay=1,
        max_cycle=2,
        nNyquist=16.0,
        max_level=7,
    )

    npt.assert_allclose(
        np.corrcoef(sample_data.real.data.ravel(), reX.sum("level").real.data.ravel()),
        np.ones((2, 2)) * 0.99,
        atol=0.01,
    )
