"""
Functions for multi-resolution dynamic-mode decomposing xarray data.
"""

import numpy as np
import xarray as xr
import scipy.linalg as spl

from .valid_coords import check_valid_dmd_coords as _check_valid_dmd_coords
from .valid_coords import get_coordinate_spacing as _get_coordinate_spacing
from .basicDMD import svht, Amatrix

__all__ = ["mrdmd"]


def _mrmodes(
    da,
    dim=None,
    spacing_tol=1e-3,
    rank=None,
    method=None,
    delay=1,
    max_cycle=2,
    nNyquist=16.0,
    sparse=False,
):
    """
    Multi-resolution DMD modes.
    """
    all_dim = list(da.dims)

    if dim is None:
        dim = all_dim
    else:
        if isinstance(dim, str):
            dim = [dim]

    _check_valid_dmd_coords(da, dim)

    axis_num = [da.get_axis_num(d) for d in dim]
    not_dim = [all_dim[n] for n in range(da.ndim) if n not in axis_num]
    M = [da.shape[n] for n in axis_num]
    # delta_t = [_get_coordinate_spacing(da[d], spacing_tol) for d in dim]

    # T = np.array(M) * np.array(delta_t)
    # cutoff_freq = max_cycle / np.prod(T)
    # sub_sample = np.ceil(1. / (cutoff_freq*nNyquist*2*np.pi*np.prod(delta_t)))
    # cutoff_freq /= sub_sample

    # time bin size
    bin_size = np.prod(np.array(M))
    if bin_size < nNyquist:
        raise ValueError(
            "Subsampled size of data is getting too small due to too many layers."
        )

    # extract subsamples
    sub_sample = bin_size // nNyquist  # max step size to capture cycles
    cutoff_freq = max_cycle / bin_size

    da_stacked = da.stack(zeta=not_dim).transpose()
    isub = [np.arange(0, M[n], sub_sample, dtype=int) for n in axis_num]
    isub = dict(zip(dim, isub))

    Xsub = da_stacked.isel(isub)  # subsampled
    # Xsub = da_stacked  # not subsampled
    X = Xsub.isel({dim[0]: slice(None, -int(delay + 1))})
    Xstacked = X.copy()
    if delay > 0:  # time-delayed coordinates
        for tt in range(int(delay)):
            if tt < delay:
                Xstacked = xr.concat(
                    [
                        Xstacked,
                        xr.DataArray(
                            Xsub.isel(
                                {dim[0]: slice(tt + 1, -int(delay + 1) + tt + 1)}
                            ).data,
                            dims=X.dims,
                            coords=X.coords,
                        ),
                    ],
                    "zeta",
                )
            else:
                Xstacked = xr.concat(
                    [
                        Xstacked,
                        xr.DataArray(
                            Xsub.isel({dim[0]: slice(int(-delay), None)}).data,
                            dims=X.dims,
                            coords=X.coords,
                        ),
                    ],
                    "zeta",
                )
    # if delay > 0:
    #     for tt in range(int(delay)):
    #         if tt < delay:
    #             Xstacked = xr.concat([Xstacked,
    #                                   xr.DataArray(Xsub.isel({dim[0]: slice(tt+1, -(delay+1)+tt+1)}).data,
    #                                        dims=X.dims, coords=X.coords
    #                                               )
    #                              ], "time")
    #         else:
    #             Xstacked = xr.concat([Xstacked,
    #                                   xr.DataArray(Xsub.isel({dim[0]: slice(delay, None)}).data,
    #                                        dims=X.dims, coords=X.coords
    #                                               )
    #                              ], "time")

    Xp = Xstacked.isel({dim[0]: slice(1, None)})

    if rank == None:
        r = svht(
            Xsub,
        )
    else:
        r = rank
    S, V, Atilde = Amatrix(Xstacked, dim, rank=r, method=method)
    Vh = np.conj(V.T)

    lamb, W = spl.eig(Atilde)
    Wh = np.conj(W.T)
    Phi = Xp.data @ V @ spl.inv(np.diag(S)) @ W

    ## Compute power of modes
    if sparse:
        # Vand_ = np.zeros((len(lamb), len(Xp[dim[0]])), dtype=np.complex128)  # Vandermonde matrix
        # for k in range(N[1]):
        #     Vand_[:,k] = lamb ** k
        # N = Xp.shape
        Vand = np.vander(lamb, len(Xp[dim[0]]), increasing=True)
        # assert np.testing.assert_array_almost_equal(Vand_[1], Vand[1])
        Vandh = np.conj(Vand.T)
        ### Algorithm based on Jovanovic et al. (2014) ##
        ### https://doi.org/10.1063/1.4863670 ###########
        G = np.diag(S) @ Vh
        Gh = np.conj(G.T)
        P = (Wh @ W) * np.conj(Vand @ Vandh)
        Pl = spl.cholesky(P, lower=True)
        q = np.conj(np.diag(Vand @ Gh @ W))
        b = spl.pinv(np.conj(Pl.T)) @ (spl.pinv(Pl) @ q)
    else:
        x0 = Xstacked.isel({dim[0]: 0}).data
        b = spl.pinv(Phi) @ x0

    # omega = np.log(lamb) / (sub_sample*np.prod(delta_t))  # Because data is subsampled, delta_t increases
    omega = np.log(lamb) / sub_sample

    omega = omega[np.nonzero(np.abs(omega) <= cutoff_freq * 2.0 * np.pi)]
    b = b[np.nonzero(np.abs(omega) <= cutoff_freq * 2.0 * np.pi)]
    Phi = Phi[..., np.nonzero(np.abs(omega) <= cutoff_freq * 2.0 * np.pi)][:, 0]

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
                "zeta": Xstacked.zeta,
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


def _mrreconstruct(
    da,
    dim=None,
    spacing_tol=1e-3,
    rank=None,
    method=None,
    delay=1.0,
    max_cycle=2,
    nNyquist=16.0,
    sparse=False,
):
    """
    Reconstruction via mrDMD.
    """
    all_dim = list(da.dims)

    if dim is None:
        dim = all_dim
    else:
        if isinstance(dim, str):
            dim = [dim]

    axis_num = [da.get_axis_num(d) for d in dim]

    Phi, omega, b = _mrmodes(
        da,
        dim=dim,
        spacing_tol=spacing_tol,
        rank=rank,
        delay=delay,
        method=method,
        max_cycle=max_cycle,
        nNyquist=nNyquist,
        sparse=sparse,
    )

    time_dynamics = b * np.exp(omega * da[dim[axis_num[0]]])

    da_recon = Phi.data.T @ time_dynamics.data

    da_recon = xr.DataArray(da_recon.T, dims=da.dims, coords=da.coords)

    return da_recon


def mrdmd(
    da,
    dim=None,
    spacing_tol=1e-3,
    rank=None,
    method=None,
    delay=1.0,
    max_cycle=2,
    nNyquist=16.0,
    max_level=8,
    sparse=False,
):
    """
    Applying DMD recursively.

    Parameters
    ----------
    da : xarray.DataArray
        The data to DMD.
    dim : str
        Dimension over SVD is taken.
    rank : int, optional
        Rank to truncate SVD.
    method : str, optional
        Method of DMD.
    delay : int
        Number of data points to stagger to capture autocorrelated features.
    max_cycle: int
        Maximum number of mode oscillations in any given time scale that qualify as "slow".
    nNyquist: float
        Non-dimensional Nyquist limit to capture cycles. Should be a product of `max_cycle`.
    max_level: int
        Maximum number of levels.
    sparse: bool, optional
        Whether to use sparse-enhanced DMD.

    Returns
    -------
    da_recon: xarray.DataArray
        Reconstruction of da on each level.
    Phi : xarray.DataArray
        DMD modes.
    omega : xarray.DataArray
        DMD frequencies.
    b: ndarray
        DMD amplitudes.
    """

    if nNyquist % max_cycle != 0.0:
        raise ValueError("`nNyquist` should be a product of `max_cycle`.")

    all_dim = list(da.dims)

    if dim is None:
        dim = all_dim
    else:
        if isinstance(dim, str):
            dim = [dim]

    axis_num = [da.get_axis_num(d) for d in dim]
    # not_dim  = [all_dim[n] for n in range(da.ndim) if n not in axis_num]
    M = [da.shape[n] for n in axis_num]

    for l in range(max_level):
        if l == 0:
            Phi, Omega, B = _mrmodes(
                da,
                dim=dim,
                spacing_tol=spacing_tol,
                rank=rank,
                method=method,
                max_cycle=max_cycle,
                nNyquist=nNyquist,
                sparse=sparse,
            )
            Phi.coords["level"] = ("mode", np.zeros(len(Phi.mode)))
            Omega.coords["level"] = ("mode", np.zeros(len(Omega.mode)))
            B.coords["level"] = ("mode", np.zeros(len(B.mode)))
            da_recon = _mrreconstruct(
                da,
                dim=dim,
                spacing_tol=spacing_tol,
                rank=rank,
                method=method,
                delay=delay,
                max_cycle=max_cycle,
                nNyquist=nNyquist,
                sparse=sparse,
            )
            da_recon.coords["sub_mode"] = (dim[0], np.ones(M[0]) * l)

            # remove influence of slow modes
            da_res = da - da_recon

        else:
            dM = M[0] // (2**l)

            for ll in range(int(2**l)):
                da_seg = da_res.isel({dim[0]: slice(dM * (ll), dM * (ll) + dM)})

                phi, omega, b = _mrmodes(
                    da_seg,
                    dim=dim,
                    rank=rank,
                    method=method,
                    delay=delay,
                    max_cycle=max_cycle,
                    nNyquist=nNyquist,
                    sparse=sparse,
                )

                phi.coords["level"] = ("mode", np.ones(len(phi.mode)) * l)
                omega.coords["level"] = ("mode", np.ones(len(omega.mode)) * l)
                b.coords["level"] = ("mode", np.ones(len(b.mode)) * l)

                Phi = xr.concat([Phi, phi], "mode")
                Omega = xr.concat([Omega, omega], "mode")
                B = xr.concat([B, b], "mode")

                recon_tmp = _mrreconstruct(
                    da_seg,
                    dim=dim,
                    spacing_tol=spacing_tol,
                    rank=rank,
                    method=method,
                    delay=delay,
                    max_cycle=max_cycle,
                    nNyquist=nNyquist,
                    sparse=sparse,
                )

                recon_tmp.coords["sub_mode"] = ("time", np.ones(dM) * ll)

                if ll == 0:
                    da_recon_tmp = recon_tmp
                else:
                    da_recon_tmp = xr.concat([da_recon_tmp, recon_tmp], dim[0])
                del recon_tmp, phi, omega, b

            # da_recon_tmp.coords['level'] = (dim[0], np.ones(len(da_recon_tmp[dim[0]]))*l)

            da_recon = xr.concat([da_recon, da_recon_tmp], "level")

            # remove influence of slow modes
            da_res = da_res - da_recon_tmp
            del da_recon_tmp

    da_recon.coords["level"] = np.arange(max_level)

    return da_recon, Phi, Omega, B
