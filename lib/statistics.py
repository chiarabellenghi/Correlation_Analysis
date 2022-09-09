# -*-coding:utf8-*-

from __future__ import print_function

# SciPy
import numpy as np
from numpy.lib.recfunctions import append_fields, drop_fields
from scipy.stats import percentileofscore
from scipy.special import erf, erfinv

from joblib import Parallel, delayed

comb = "combined"


def pVal(N_obs, N_trials, n_boots=1000):
    assert(N_obs.dtype.names == N_trials.dtype.names)

    names = N_obs.dtype.names
    if len(names) > 1:
        if comb in names:
            raise ValueError("Sample with name " + comb + " already found")
        names += comb,

    pV = np.ones(N_obs.shape, dtype=[(k, np.float32) for k in names])

    M = np.empty(N_trials.shape, dtype=np.bool)

    for k in N_obs.dtype.names:
        np.greater_equal(N_trials[k], N_obs[k], out=M)
        pV[k] = M.sum(axis=0, dtype=np.float32) / len(N_trials)

    if comb in names:
        for k in N_obs.dtype.names:
            pV[comb] *= pV[k]

    if n_boots < 1:
        return pV

    pV_errs = np.array([pVal(N_obs, N_trials[np.random.choice(len(N_trials),
                                                              np.random.poisson(len(N_trials)))],
                             n_boots=0)
                        for i in range(n_boots)])

    if pV.ndim == 1:
        pV_errs = np.vstack(pV_errs)
    else:
        pV_errs = np.vstack(pV_errs[np.newaxis])

    pV_err = np.empty((2,) + pV.shape, dtype=pV.dtype)

    for k in pV_err.dtype.names:
        pV_err[k] = np.percentile(pV_errs[k], [15.87, 84.13], axis=0)

    return pV, pV_err


def pVal_trial(N_trials, n_trials, idx):
    '''Perform p-value calculation on trials for post-trial calculation.

    '''

    m = np.ones(n_trials, dtype=np.bool)
    m[idx] = False

    N_cp = N_trials[idx]
    N_trials = N_trials[m]

    return pVal(N_cp, N_trials, n_boots=0)


def calc_pVal_trials(Ntrials):

    n_trials = len(Ntrials)

    shape = Ntrials.shape
    if len(shape) > 2:
        tmpshape = shape[:1] + (np.prod(shape[1:]), )

        # flatten n-dimensional array and call function, reshape output
        return calc_pVal_trials(Ntrials.reshape(tmpshape)).reshape(shape)

    names = Ntrials.dtype.names
    if len(names) > 1:
        if comb in names:
            raise ValueError("Existing field", comb)
        names += (comb, )

    pVal = np.empty(shape, dtype=[(k, np.float32) for k in names])

    for k in Ntrials.dtype.names:
        N = np.amax(Ntrials[k])

        P = np.vstack([np.bincount(Ni, minlength=N+1)
                      for Ni in Ntrials[k].T]).T

        P = P[::-1].cumsum(axis=0)[::-1].astype(np.float32) / n_trials

        row = np.repeat(
            np.arange(Ntrials[k].shape[-1])[np.newaxis], n_trials, axis=0)

        pVal[k] = P[Ntrials[k], row]

    if comb in names:
        pVal[comb] = np.prod([pVal[k] for k in Ntrials.dtype.names], axis=0)

    return pVal


def calc_pVal_trials_mem(Ntrials):
    '''calc_pVal_trials version with less memory consumption

    '''

    n_trials = len(Ntrials)

    shape = Ntrials.shape
    if len(shape) > 2:
        tmpshape = shape[:1] + (np.prod(shape[1:]), )

        return calc_pVal_trials_mem(Ntrials.reshape(tmpshape)).reshape(shape)

    names = Ntrials.dtype.names
    if len(names) > 1:
        if comb in names:
            raise ValueError("Existing field", comb)
        names += (comb, )

    # create output array
    pVal = np.empty(shape, dtype=[(k, np.float32) for k in names])

    for k in Ntrials.dtype.names:
        for i in np.arange(shape[-1]):
            Ni = Ntrials[k][:, i]

            P = np.cumsum(np.bincount(Ni, minlength=Ni.max() + 1)
                          [::-1], dtype=np.float32)[::-1]
            P /= n_trials

            pVal[k][:, i] = P[Ni]

    if comb in names:
        pVal[comb] = np.prod([pVal[k] for k in Ntrials.dtype.names], axis=0)

    return pVal


def post_trial(NpV, Ntrials, MpV=None, Mtrials=None, idx=0):
    # primaries
    n_trials = len(Ntrials)

    assert(NpV.dtype.names == Ntrials.dtype.names)

    snd_idx = dict()

    # secondaries
    if (MpV is not None) ^ (Mtrials is not None):
        raise ValueError("Need both data and trials for secondary data")

    if (MpV is not None) & (Mtrials is not None):
        m_trials = len(Mtrials)

        assert(m_trials == n_trials)

        assert(MpV.dtype.names == Mtrials.dtype.names)

        # get correlated fields, disregard other ones (for now)
        MpV = MpV[idx]
        Mtrials = Mtrials[:, idx]

        # prepare output
        fields = ([k for k in Ntrials.dtype.names if not k == comb]
                  + [k for k in Mtrials.dtype.names if not k == comb]
                  + ([comb + "_primary"] if comb in Ntrials.dtype.names else [])
                  + ([comb + "_secondary"] if comb in Mtrials.dtype.names else [])
                  + [comb])

        data = np.ones(NpV.shape, dtype=[(k, np.float32) for k in fields])
        trials = np.ones(Ntrials.shape, dtype=[
                         (k, np.float32) for k in fields])

        for k in Ntrials.dtype.names:
            data[k] *= NpV[k]
            trials[k] *= Ntrials[k]
            if k == comb:
                data[comb + "_primary"] *= NpV[k]
                trials[comb + "_primary"] *= Ntrials[k]

        for k in Mtrials.dtype.names:
            data[k] *= np.amin(MpV[k], axis=0)
            trials[k] *= np.amin(Mtrials[k], axis=1)
            snd_idx[k] = np.argmin(Mtrials[k], axis=1)
            if k == comb:
                data[comb + "_secondary"] *= np.amin(MpV[k], axis=0)
                trials[comb + "_secondary"] *= np.amin(Mtrials[k], axis=1)
                snd_idx[k + "_secondary"] = np.argmin(Mtrials[k], axis=1)

        if comb not in Ntrials.dtype.names:
            data[comb] *= np.prod([NpV[k] for k in NpV.dtype.names], axis=0)
            trials[comb] *= np.prod([Ntrials[k]
                                    for k in Ntrials.dtype.names], axis=0)

        if comb not in Mtrials.dtype.names:
            data[comb] *= np.amin(np.prod([MpV[k] for k in MpV.dtype.names], axis=0),
                                  axis=0)
            trials[comb] *= np.amin(np.prod([Mtrials[k] for k in Mtrials.dtype.names], axis=0),
                                    axis=1)
    else:
        data = NpV
        trials = Ntrials

    min_pVals = np.empty(len(trials), dtype=trials.dtype)
    min_idx = np.empty(len(trials), dtype=[(k, np.int)
                                           if k not in snd_idx
                                           else (k, int, 2)
                                           for k in min_pVals.dtype.names])
    for k in min_pVals.dtype.names:
        min_pVals[k] = np.amin(trials[k], axis=1)
        fst_idx = np.argmin(trials[k], axis=1)
        min_idx[k] = fst_idx if k not in snd_idx else np.vstack(
            (fst_idx, np.choose(fst_idx, snd_idx[k].T))).T

        m = ~min_pVals[k].astype(np.bool)

        min_pVals[k][m] = 1. / (n_trials + 2)

    post = dict()
    ppval = dict()

    for k in min_pVals.dtype.names:
        ppval[k] = data[k].min()
        post[k] = percentileofscore(
            min_pVals[k], data[k].min(), kind="weak") / 100.

        if not bool(post[k]):
            print("Warning: for {0:s}, no trials more significant".format(k))
            post[k] = 1. / (n_trials + 2)

    return post, ppval, min_pVals, min_idx


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
