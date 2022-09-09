#!/usr/bin/python

import numpy as np
import healpy as hp
from utils.math_utility import get_pix2ang, GreatCircleDistance


class LocalHotSpotsList(object):
    hotspot_dtype = [("dec", float), ("ra", float), ("logpVal", float)]

    def __init__(self):
        self._hotspot_list = np.recarray((0,), dtype=self.hotspot_dtype)

    def add(self, dec, ra, logpVal):
        r"""Add local hotspots to the list.

        Parameters
        ----------
        dec: float | array of float
            Declination(s) in radians.
        ra: float | array of float
            Right Ascension(s) in radians
        logpVal: float | array of float
            Significance(s) in -log10(p-value)
        """

        # Add new hotspots to the list.
        spots = np.empty(dec.size, dtype=self.hotspot_dtype)
        spots["dec"] = dec
        spots["ra"] = ra
        spots["logpVal"] = logpVal
        self._hotspot_list = np.concatenate([self._hotspot_list, spots])
        self._hotspot_list.sort(order="logpVal")

    def stack(self, spots):
        # Stack two lists of hotspots.
        self._hotspot_list = np.concatenate([self._hotspot_list, spots])
        self._hotspot_list.sort(order="logpVal")

    @property
    def list(self):
        return self._hotspot_list

    @list.setter
    def list(self, value):
        self._hotspot_list = value


class skyscan_handler(object):
    r"""Deal with all sky scans produced for the improved PS analysis."""

    def __init__(self, path=None, nside=256):
        r"""
        Parameters
        ----------
        path : str | None
            Path to the HealPy map.
        nside : int
            Resolution parameter of the HealPy map.
        """

        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.skyscan_dtype = [("npix", int), ("log10p", float)]
        self.full_skyscan = np.empty((self.npix,), dtype=self.skyscan_dtype)

        # For the HPA, we always use the -log10(p-value)
        if path is not None:
            scan = np.load(path)
            self._log10p_map = -np.log10(scan["pvalue"])
            self._pix_map = scan["npix"]
        else:
            self._log10p_map = None
            self._pix_map = None

    @property
    def log10p_map(self):
        return self._log10p_map

    @log10p_map.setter
    def log10p_map(self, value):
        self._log10p_map = value

    @property
    def pix_map(self):
        return self._pix_map

    @pix_map.setter
    def pix_map(self, value):
        self._pix_map = value

    def prepare_full_scan(self, dec_range=None):
        r"""Only analyze from -3 to 81 deg in declination. But a full skymap
        is useful for the hotspot search. Fill the missing pixels with zeros.

        N.B.: If a declination range is chosen, every -log10(p-value) ouside
        it range will be set to zero. If it is larger than the simulated one,
        then the simulated range is used. The full skyscan is always built.
        """

        full_sky_pvalues = np.zeros(self.npix)
        if dec_range is None:
            full_sky_pvalues[self._pix_map] = self._log10p_map
        else:
            # Check whether the declination range is covered in the pixel map.
            min_dec, max_dec = np.radians(dec_range[0]), np.radians(
                dec_range[1]
            )
            dec, _ = get_pix2ang(self.nside, self._pix_map)
            if np.radians(dec_range[0]) < np.min(dec):
                min_dec = np.min(dec)
            if np.radians(dec_range[1]) > np.max(dec):
                max_dec = np.max(dec)

            mask_pix = hp.query_strip(
                self.nside,
                np.pi / 2 - max_dec,
                np.pi / 2 - min_dec,
            )
            mask = np.in1d(self._pix_map, mask_pix)
            full_sky_pvalues[mask_pix] = self._log10p_map[mask]

        self._full_skyscan["npix"] = np.arange(self.npix)
        self._full_skyscan["log10p"] = full_sky_pvalues

    @property
    def full_skyscan(self):
        return self._full_skyscan

    @full_skyscan.setter
    def full_skyscan(self, value):
        if value.dtype != self.skyscan_dtype:
            raise NotImplementedError()
        self._full_skyscan = value

    def get_hotspots(self, log10p_threshold, psi_min=None):
        """Find all pvalues above some threshold. Return an array of warm
        spots having the same `dtype` of the skyscan record array.
        """
        skyscan = self._full_skyscan
        log10p = skyscan["log10p"]
        m = log10p > log10p_threshold

        npix = len(log10p)
        nside = hp.npix2nside(npix)

        sel_pix = np.arange(npix)[m]

        warm_pix = []
        for pix in sel_pix:
            neighbours = hp.get_all_neighbours(nside, pix)
            # Only check closest neighbours that are in the selected sky points
            mask = np.asarray(
                [neighbour in sel_pix for neighbour in neighbours]
            )
            neighbours = neighbours[mask]
            # If none of the closest_neighbours p-value is larger than the
            # one of the pixel we're looking at, we have a warm spot.
            if not any(log10p[neighbours] > log10p[pix]):
                warm_pix.append(pix)

        dec, ra = get_pix2ang(nside, warm_pix)
        logpVal = log10p[warm_pix]

        spots = LocalHotSpotsList()
        spots.add(dec, ra, logpVal)

        if psi_min is not None:
            hotspots = self.cut_close_spots(spots, psi_min)
        else:
            hotspots = spots.list

        return hotspots

    @staticmethod
    def cut_close_spots(spots, psi_min):
        # The skyscan must be sorted by -log10(p-value)!
        spots.list.sort(order="logpVal")

        idx_to_delete = []
        for i in np.arange(spots.list.size):
            ang_dist = GreatCircleDistance(
                spots.list["ra"][i],
                spots.list["dec"][i],
                spots.list["ra"][i + 1:],
                spots.list["dec"][i + 1:],
            )
            mask = np.where(ang_dist < np.radians(psi_min))[0]
            if len(mask) == 0:
                continue
            if any(
                spots.list["logpVal"][mask + i + 1] >= spots.list["logpVal"][i]
            ):
                idx_to_delete.append(i)
        mask = np.logical_not(np.in1d(range(spots.list.size), idx_to_delete))
        return spots.list[mask]
