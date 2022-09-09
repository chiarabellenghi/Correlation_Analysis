import numpy as np
import healpy as hp
from scipy.stats import poisson
from scipy.special import erf, erfinv


def GreatCircleDistance(ra_1, dec_1, ra_2, dec_2):
    r"""Compute the great circle distance between points in the sky.
    All coordinates must be given in radians.
    """
    delta_dec = np.abs(dec_1 - dec_2)
    delta_ra = np.abs(ra_1 - ra_2)
    x = (np.sin(delta_dec / 2.0)) ** 2.0 + np.cos(dec_1) * np.cos(dec_2) * (
        np.sin(delta_ra / 2.0)
    ) ** 2.0
    return 2.0 * np.arcsin(np.sqrt(x))


def get_pix2ang(nside, pix):
    r"""Compute the declination and right ascension corresponding
    to the given healpy pixel. Both are returned in radians.

    Parameters
    ----------
    nside: int
        Resolution parameter for the Healpy grid.
    pix: int
        Healpy pixel to be converted into angular coordinates.

    Returns
    -------
        Declination and RA in radians.

    """
    co_lat, lon = hp.pix2ang(nside, pix, lonlat=False)
    return np.pi / 2 - co_lat, lon


def poisson_weights(vals, mean, weights=None):
    r"""Calculate weights for a sample so that it follows a Poisson.

    Parameters
    ----------
    vals : array_like
        Random integers to be weighted.
    mean : float
        Poisson mean.
    weights : array_like, optional
        Weights for each event.

    Returns
    -------
    ndarray
        Weights for each event.
    """

    mean = float(mean)
    vals = np.asarray(vals, dtype=int)

    if weights is None:
        weights = np.ones_like(vals, dtype=float)

    # Get occurences of integers.
    # (Lenght of the output of numpy.bincount(x) is equal to np.amax(x)+1.)
    bincount = np.bincount(vals, weights=weights)
    n_max = bincount.size

    # Get poisson probability.
    if mean > 0:
        p = poisson(mean).pmf(range(n_max))
    else:
        p = np.zeros(n_max, dtype=float)
        p[0] = 1.0

    # Weights for each integer.
    w = np.zeros_like(bincount, dtype=float)
    m = bincount > 0
    w[m] = p[m] / bincount[m]

    w = w[np.searchsorted(np.arange(n_max), vals)]

    return w * weights


def poisson_percentile(mu, x, y, yval):
    r"""Calculate upper percentile using a Poisson distribution.

    Parameters
    ----------
    mu : float
        Mean value of Poisson distribution.
    x : array_like,
        Trials of variable that is expected to be Poisson distributed.
    y : array_like
        Observed variable connected to `x`.
    yval : float
        Value to calculate the percentile at.

    Returns
    -------
    score : float
        Value at percentile *alpha*
    err : float
        Uncertainty on `score`
    """

    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=float)

    w = poisson_weights(x, mu)

    # Get percentile at yval.
    m = y > yval
    u = np.sum(w[m], dtype=float)

    if u == 0.0:
        return 1.0, 1.0

    err = np.sqrt(np.sum(w[m] ** 2)) / np.sum(w)

    return u / np.sum(w, dtype=float), err, w


def sigma2pval(sigma, oneSided=False):
    r"""Converts gGaussian sigmas into p-values.

    Parameters
    ----------
    sigma : float or array_like
        Gaussian sigmas to be converted in p-values
    oneSided: bool, optional, default=False
        If the sigma should be considered one-sided or two-sided.

    Returns
    -------
    pval : array_like
        p-values.
    """

    pval = 1 - erf(sigma / np.sqrt(2))
    if oneSided:
        pval /= 2.0
    return pval


def pval2Sigma(pval, oneSided=False):
    r"""Converts p-values into Gaussian sigmas.
    Parameters
    ----------
    pval : float or array_like
        p-values to be converted in Gaussian sigmas
    oneSided: bool
        If the sigma should be calculated one-sided or two-sided.
        Default: False.
    Returns
    -------
    ndarray
        Gaussian sigma values
    """
    if oneSided:
        pval *= 2.0
    sigma = erfinv(1.0 - pval) * np.sqrt(2)
    return sigma


def construct_mu_per_flux(
    mc, livetime=3186.105475562, dec=None, gamma=2.0, refflux_E0=1
):
    r"""This method is meant to calculate the factor to convert from number
    of injectedneutrino events to flux normalisation and viceversa.

    Parameters
    ----------
    mc: structured array
        The structured array containing Monte Carlo data.
    livetime : float
        Detector livetime in days. Default: Livetime taken from
        IC86 2011-2019 GRL.
    dec : float | list of two float | None
        Needed to select a declination band to calculate the conversion factor.
        If float, it is the declination of the injection point and the factor is
        computed taking MC data in a 2 degrees bandwidth around that declination.
        if a list of 2 float, that is the declination bandwidth that will be
        used to compute the conversion factor.
        If None, the default declination range for this analysis is used,
        i.e. (-3, 81) deg. Argument must be given in degrees.
    refflux_E0 : float
        Reference energy to use for the power law flux model. Default: 1 GeV.

    Returns
    -------
    mu_per_flux : float
    """

    # Detector livetime in seconds.
    T = livetime * 24 * 60 * 60

    if dec is not None:
        if isinstance(dec, float):
            sin_dec_bandwidth = np.sin(np.radians(1))
            min_sin_dec = np.sin(np.radians(dec)) - sin_dec_bandwidth
            max_sin_dec = np.sin(np.radians(dec)) + sin_dec_bandwidth
        elif len(dec) == 2 and all(isinstance(d, (int, float)) for d in dec):
            max_sin_dec = np.sin(np.radians(max(dec)))
            min_sin_dec = np.sin(np.radians(min(dec)))
        else:
            raise RuntimeError("Error to be implemented.")
    else:
        max_sin_dec = np.sin(np.radians(81))
        min_sin_dec = np.sin(np.radians(-3))

    # Mask the MC
    mask = (np.sin(mc["trueDec"]) < max_sin_dec) & (
        np.sin(mc["trueDec"]) > min_sin_dec
    )
    tmc = mc[mask]

    # Solid angle for the effective area calculation.
    delta_sin_dec = max_sin_dec - min_sin_dec
    solidAngle = 2 * np.pi * delta_sin_dec

    mu_per_flux = (
        np.sum(tmc["ow"] * (tmc["trueE"] / refflux_E0) ** (-abs(gamma)))
        * T
        / (solidAngle)
    )

    return mu_per_flux
