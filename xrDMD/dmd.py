"""
Functions for dynamic-mode decomposing xarray data.
"""

import numpy as np
import xarray as xr
import scipy.linalg as spl


def Amatrix(X, rank, method=None):
    """
        Perform SVD decomposition to obtain A.

        Parameters
    ----------
    da : xarray.DataArray
        The data to DMD
    dim : str or list
        Dimensions along which to apply detrend.
        Can be either one dimension or a list with two dimensions.
        Higher-dimensional detrending is not supported.
        If dask data are passed, the data must be chunked along dim.

    Returns
    -------
    da : xarray.DataArray
        The detrended data.
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
