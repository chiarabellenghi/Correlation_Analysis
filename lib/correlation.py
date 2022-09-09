# -*-correlation:utf8-*-

# SciPy
import numpy as np

# Healpy
import healpy as hp

from joblib import Parallel, delayed

# local
import lib.core as core


class Correlator(object):
    '''Main class that correlates objects with a catalogue.

    Parameters
    -----------
    catalogue : structured array
        Catalogue arrays with important fields: ra, dec, FOM.
    bins : array
        Bins to use for figure of merit.

    Optional Parameters
    --------------------
    mlat : float
        Value in radians to cut the galactic plane in random scrambling.
    dec_range : float
        Sequence of two values for the minimum and maximum declination defining
        the portion of the sky to be analysed.
    seed : int
        Seed for the randomization.

    Attributes
    --------------------
    primaries : dict()
        Structure to store primaries to correlate with sources (neutrinos).
    secondaries : dict()
        Structure to store secondaries to correlate with sources selected by
        primaries (CR).
    mcat : ndarray(shape: (len(bins), len(catalogue))
        Mask that stores the information about which bin contains which sources.
    catalogue : structured array
        As input, sources that are in no bin are removed.
    v : ndarray
        Stores the position of the sources as 3D vectors.
    random :
        Numpy RandomState() initialized with seed.
    boots : int

    scramble : int
        Set the type of scrambling to apply. See self.scramble definition for
        details.
    '''
    _mlat = np.radians(10.)
    _v_scramble = None
    _u_scramble = None
    _boots = None
    _seed = None
    _hemisphere = None
    _logpVal_thr = None
    _dec_range = (np.radians(-90.), np.radians(90.))

    def __init__(self, catalogue, bins, seed=None, boots=None, **kwargs):

        for key, val in kwargs.items():
            if not hasattr(self, key):
                raise ValueError("Parameter " + key + " unknown")

            setattr(self, key, val)

        self.primaries = {}
        self.secondaries = {}

        print("Catalogue: {0:d} sources".format(len(catalogue)))
        self._boots = boots

        self.bins = []
        self.mcat = []

        print("Applying cuts to the source catalogue:")
        # Cut the hemisphere.
        mask_hemi = np.logical_and(
            catalogue['dec'] >= self.dec_range[0],
            catalogue['dec'] <= self.dec_range[1]
        )
        catalogue = catalogue[mask_hemi]
        print(
            f"\tSources in declination range: {np.rad2deg(self.dec_range)} "
            f"degrees: {len(catalogue)}.")

        # Cut the galactic plane if required.
        if self.mlat is not None:
            print("\tApplying galactic plane cut...")
            _, lat = core.galactic(
                catalogue['ra'], catalogue['dec']
            )
            mask = np.fabs(lat) > self.mlat
            catalogue = catalogue[mask]
            print(
                f"\tAfter galactic plane cut: {len(catalogue)}")

        for b_i in np.sort(bins):
            mc = catalogue["FOM"] >= b_i

            if len(self.mcat) < 1 or np.any(mc != self.mcat[-1]):
                self.bins.append(b_i)
                self.mcat.append(mc)

                print("\t{0:4d} sources above {1:.2f} ({2:7.2%})".format(
                    mc.sum(), b_i, mc.sum(dtype=np.float) / len(mc)))
            else:
                print("\tno new sources at {0:.2f}, skip".format(b_i))

        self.bins = np.asarray(self.bins)
        self.mcat = np.asarray(self.mcat)

        mcat = np.any(self.mcat, axis=0)
        print("Remove sources that are in no bin: {0:7.2%}".format(
            (~mcat).sum(dtype=np.float) / len(mcat)))
        self.catalogue = catalogue[mcat]
        self.mcat = self.mcat.T[mcat].T

        # Calculate the vectors for the sources
        self.v = core.ang2vec(self.catalogue["ra"], self.catalogue["dec"])

        if seed is None:
            print("Analysis - Initialize with random seed")
            self.random = np.random.RandomState()
        else:
            print(f"Analysis - Initialize with seed {seed}")
            self.random = np.random.RandomState(seed)

        return

    def __call__(self, scramble=False, max_shift=None):
        '''Calculate the number of counterparts between sources and primaries.

        Parameters
        ----------
        scramble : bool
            Scrambles the position of sources/primaries.

        Returns
        ---------
        N_cp : ndarray(shape: (# of bins,))
            Number of events with one counterpart.
        '''
        v_scramble, u_scramble = self.scramble if scramble else (False, False)

        v, mcat = self.get_v(scramble=v_scramble, max_shift=max_shift)

        if len(self.primaries) < 1:
            raise ValueError("No primary samples inserted")

        N_cp,  = self._get_cp(v, mcat, u_scramble)[0]

        return N_cp

    def _get_cp(self, v, mcat, scramble):
        '''Calculates the number of counterparts between v and primaries

        Parameters
        ----------
        v : array
            Array containing the 3D-vector positions of the Sources
        mcat : ndarray
            Mask that stores the information about which bin contains which sources.
        scramble : bool
            Scrambling the primaries

        Returns
        ----------
        N_cp : ndarray
            Number of events with one counterpart.
        m_cp : mask
            Mask with sources that are counterparts for at least one neutrino.
        '''
        m_cp = np.zeros_like(mcat)
        N_cp = np.empty(len(mcat), dtype=[(k, np.uint16)
                        for k in self.primaries])

        for key, sam in self.primaries.items():
            u, sigma = sam.get_vector(scramble=scramble)
            # print("---")
            # print("Scramble {0}".format(scramble))
            # for i in u:
            #    print("{0} -  {1} -  {2}".format(i[0], i[1], i[2]))
            # print("---")

            # calculate distance of events and catalogue
            # cosD : ndarray(shape: (# neutrinos, # sources))
            cosD = core.dist(u, v)

            # select sources within error circle
            # m : mask(shape like cosD)
            m = cosD >= np.cos(sigma)[:, np.newaxis]

            # number of correlations: FOM vs events vs sources
            # M : mask(shape: (# bins, #neutrinos, # sources)
            M = m[np.newaxis] & mcat[:, np.newaxis]

            # add sources that correlate with at least one event,
            # m_cp : mask(shape: (# bins, # sources))
            # mask with sources that have at least one neutrino
            m_cp |= np.any(M, axis=1)

            # count number of events with one counterpart
            # N_cp[key] : ndarray(shape : (# bins,)
            N_cp[key] = np.sum(np.any(M, axis=2), axis=1, dtype=np.uint16)

        return N_cp, m_cp

    @property
    def boots(self):
        return self._boots

    @boots.setter
    def set_boots(self, value):
        '''Define scrambling scheme to apply to the catalogue.

        Parameters
        ----------
        value : int
            value 0 : Scramble galactic Latitude and galactic Longitude of sources,
                      do not scramble Galactic Plane.
            value 1 : Scramble galactic Longitude of sources.
            value 2 : Scramble right ascension of sources, avoid Galactic Plane.
            value 3 : Scramble right ascension and declination of sources,
                      avoid Galactic Plane.
            value 4 : Scramble Alpaka
            value 5 : Scramble Alpaka, no bands
            value 6 : Scramble right ascension and declination of sources,
                      avoid Galactic Plane. Apply a maximum shift if required.
                      Preserve number of sources in the sky.
        '''
        value = int(value)
        self._boots = value
        return

    @property
    def mlat(self):
        return self._mlat

    @mlat.setter
    def mlat(self, value):
        value = float(value)
        if value < 0 or value > np.pi / 2.:
            raise ValueError("mlat not in [0, pi /2] range")

        print(
            "- Setting galactic plane width to {0:.1f}deg".format(np.degrees(value)))

        self._mlat = value

        return

    @property
    def scramble(self):
        return self._v_scramble, self._u_scramble

    @scramble.setter
    def scramble(self, value):
        '''Define the type of scrambling to apply.

        Parameters
        ----------
        value : int
            value 0  : Scramble only sources coordinates.
            value 1  : Scramble only primaries coordinates.
            value -1 : Scramble both sources and primaries coordinates.
        '''
        value = int(value)
        if value == 0:
            print("Set scrambling to catalogue")
            self._v_scramble = True
            self._u_scramble = False
        elif value == 1:
            print("Set scrambling to primaries")
            self._v_scramble = False
            self._u_scramble = True
        elif value == -1:
            print("Scramble everything")
            self._v_scramble = True
            self._u_scramble = True
        else:
            raise ValueError("Scrambling mode none of (0, 1, -1)")

        return

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        val = int(value)
        self._seed = val
        self.random.seed(val)

        for prim in self.primaries.values():
            prim.random.seed(self.random.randint(2**32))

        return

    @property
    def dec_range(self):
        return self._dec_range

    @dec_range.setter
    def dec_range(self, value):
        if not hasattr(value, "__iter__"):
            raise TypeError(
                f"The declination range must be an iterable!"
            )

        if len(value) != 2:
            raise ValueError(
                "The declination range must be a sequence of 2 values."
            )

        # Order it and make it unmutable.
        self._dec_range = tuple(sorted(value))

        return

    @property
    def logpVal_thr(self):
        return self._logpVal_thr

    @logpVal_thr.setter
    def logpVal_thr(self, value, name='hotspots'):
        if not isinstance(value, float):
            raise ValueError("To be implemented.")

        self._logpVal_thr = value

        if name != 'hotspots':
            return

        print("\tApplying the pvalue threshold cut...")
        if 'logpVal' in self.primaries[name].events.dtype.names:
            mask = self.primaries[name].events['logpVal'] > self.logpVal_thr
            self.primaries[name].events = self.primaries[name].events[mask]
        else:
            print(
                "\n*****************************************************\n"
                "Warning! The sample is missing the 'logpVal' field. "
                "The pVal threshold cut will NOT be applied."
                "\n*****************************************************\n"
            )
        print(
            f"\tSpots with -log10p > {value}: {self.primaries[name].events.size}"
        )

        self.primaries[name].u = core.ang2vec(
            self.primaries[name].events['ra'], self.primaries[name].events['dec']
        )
        self.primaries[name].N = len(self.primaries[name].events)

        print(
            f"After the cut(s), {self.primaries[name].N} hotspots will be analysed.")

        return

    def add_primary(self, name, sample):
        '''Add EventSample to primaries

        Parameters
        ----------
        name : string
            Name of the sample.
        sample : EventSample
            Sample of neutrino events to add to the primaries dict().
        '''
        # assert(not (self.secondaries.has_key(name)
        #        or self.primaries.has_key(name)))

        assert(isinstance(sample, EventSample))

        print("Applying cuts to the event sample:")

        # Cut the hemisphere if required.
        mask_hemi = np.logical_and(
            sample.events['dec'] >= self._dec_range[0],
            sample.events['dec'] <= self._dec_range[1]
        )
        sample.events = sample.events[mask_hemi]
        print(
            f"\tEvents in {np.rad2deg(self.dec_range)} degrees: "
            f"{sample.events.size}")

        # Cut the galactic plane if required.
        if self.mlat is not None:
            print("\tApplying galactic plane cut...")
            _, ev_lat = core.galactic(
                sample.events['ra'], sample.events['dec']
            )
            mask = np.logical_or(ev_lat < -self.mlat, ev_lat > self.mlat)
            sample.events = sample.events[mask]
            print(f"\tAfter galactic plane cut: {sample.events.size}")

        # Update the attributes of the EventSample object:
        sample.u = core.ang2vec(sample.events['ra'], sample.events['dec'])
        sample.N = len(sample.events)

        print(f"After the cut(s), {sample.N} hotspots will be analysed.")

        self.primaries[name] = sample

        return

    def get_counterparts(self, binval, verbose=True, invert=False, graph=None):
        '''Return mask of counterparts for events. Not on scrambled data.

        Parameters
        ----------
        binval : array
            Bins to use for figure of merit
        verbose : bool
            Print counterparts information (default: True)
        invert : bool
            (default: False)
        graph : graph
            (default: None)
        '''
        mcat = self.catalogue["FOM"] >= binval

        m_cp = np.zeros_like(mcat)

        result = dict()

        for key, sam in self.primaries.items():
            u, sigma = sam.get_vector()

            cosD = core.dist(u, self.v)

            m = cosD >= np.cos(sigma)[:, np.newaxis]
            m_cp |= np.any(m & mcat, axis=0)

            result[key] = np.any(m & mcat, axis=1)

            if verbose:
                print(
                    f"{key}: {np.count_nonzero(result[key])} counterparts above {binval}"
                )

                if np.count_nonzero(result[key]) < 1:
                    continue

                # print("\t{0:10s} - {1:22s} - {2:5s}".format("Event logpVal",
                #       "Source", "Offset (Fraction)"))
                print(
                    "\tR.A.[deg]  -  Dec.[deg]  -  HS logpVal  -  Source Name  -  Offset[deg]  -  Redshift")
                for m_i, d_i, ev_i in zip(m & mcat, cosD, sam.events):
                    if np.all(~m_i):
                        continue
                    idx = self.catalogue[m_i]["id"]
                    zs = self.catalogue[m_i]["FOM"]
                    D = np.degrees(np.arccos(d_i[m_i]))
                    # F = (1. - d_i[m_i]) / (1. - np.cos(ev_i["sigma"]))

                    # if not invert and graph is not None:
                    #     if not hasattr(graph, "nodesize"):
                    #         graph.nodesize = {}

                    #     graph.add_node("{0:s} {1:s}".format(key, ev_i["id"]))
                    #     graph.nodesize["{0:s} {1:s}".format(
                    #         key, ev_i["id"])] = ev_i["E"]

                    s = np.argsort(idx)
                    for id, D, z in zip(idx[s], D[s], zs[s]):
                        if z > 990:
                            z = np.nan
                        print(
                            f"\t{np.degrees(ev_i['ra']):<9.2f}     "
                            f"{np.degrees(ev_i['dec']):<8.2f}      "
                            f"{ev_i['logpVal']:<10.2f}    {str(id)[2:-1]:<11s}   {D:<11.2f}     {z:<11.2f}"
                        )
                        # print("\t{0:6s} - {1:24s} - {2:5.2f} ({3:7.2%})".format(ev_i["id"], id, D, 1-f))

                        # if not invert and graph is not None:
                        #     graph.add_node(id)
                        #     graph.add_edge("{0:s} {1:s}".format(key, ev_i["id"]), id,
                        #                    weight=1-f)

                        #     graph.nodesize[id] = self.catalogue[self.catalogue["id"]
                        #                                         == id]["FOM"][0]**2

        if invert:
            m_cp = mcat & (~m_cp)

        return result, m_cp

    def get_v(self, scramble=False, max_shift=None):
        '''Return vector arrays and mcat, scramble if wanted.

        Parameters
        ----------
        scramble : bool
            Scramble source position. Three different scramblings are implemented.
            See self.boots for details.

        Returns
        ----------
        v : array
            Stores the position of the sources as 3D vectors.
        mcat : mask
            Mask that stores the information about which bin contains which sources.
        '''
        if scramble:

            n = len(self.catalogue)

            if self.boots == 0:
                # Scramble latitude and longitude, avoid plane
                lat = self.random.uniform(-1. + np.sin(self.mlat),
                                          1 - np.sin(self.mlat), n)
                lat += np.where(lat > 0., np.sin(self.mlat), -
                                np.sin(self.mlat))
                lat = np.arcsin(lat)

                lon = self.random.uniform(0., 2.*np.pi, n)

                v = core.ang2vec(*core.inv_galactic(lon, lat))
                mcat = self.mcat

            elif self.boots == 1:
                # scramble only longitude, transform back
                lon, lat = core.galactic(
                    self.catalogue["ra"], self.catalogue["dec"])

                lon = self.random.uniform(0., 2. * np.pi, n)

                v = core.ang2vec(*core.inv_galactic(lon, lat))
                mcat = self.mcat

            elif self.boots == 2:
                # Only scramble right-ascension, but not on galactic plane

                def check_plane(ra, dec):
                    lon, lat = core.galactic(ra, dec)

                    return np.fabs(lat) < self.mlat

                dec = self.catalogue["dec"]
                ra = self.random.uniform(0., 2.*np.pi, n)

                m = check_plane(ra, dec)
                while np.any(m):
                    ra[m] = self.random.uniform(
                        0., 2.*np.pi, np.count_nonzero(m))

                    m = check_plane(ra, dec)

                v = core.ang2vec(ra, dec)
                mcat = self.mcat

            elif self.boots == 3:
                # Scramble right-ascension and declination, but not on galactic plane

                def check_plane(ra, dec):
                    lon, lat = core.galactic(ra, dec)

                    return np.fabs(lat) < self.mlat

                dec = np.arcsin(self.random.uniform(-1., 1., n))
                ra = self.random.uniform(0., 2.*np.pi, n)

                m = check_plane(ra, dec)
                while np.any(m):
                    dec[m] = np.arcsin(
                        self.random.uniform(-1., 1., np.count_nonzero(m)))
                    ra[m] = self.random.uniform(
                        0., 2.*np.pi, np.count_nonzero(m))

                    m = check_plane(ra, dec)

                v = core.ang2vec(ra, dec)
                mcat = self.mcat

            elif self.boots == 4:
                # Generate source positions as Alpaka catalog

                alpaka_mask = np.load("../data/alpaka_skymask_nside64.npy")
                nside = 64

                dec = []
                ra = []

                def check_mask(ra, dec, mask, nside):
                    idx = hp.ang2pix(nside, np.pi / 2. - dec, ra)
                    return mask[idx]

                for i in np.arange(n):
                    while(True):
                        rai = np.random.uniform(0., 2*np.pi)
                        deci = np.arcsin(np.random.uniform(-1., 1.))
                        cond = check_mask(rai, deci, alpaka_mask, nside)
                        if cond:
                            ra.append(rai)
                            dec.append(deci)
                            break

                dec = np.array(dec)
                ra = np.array(ra)

                v = core.ang2vec(ra, dec)
                mcat = self.mcat

            elif self.boots == 5:
                # Generate source positions as Alpaka catalog, no bands

                alpaka_mask = np.load(
                    "../data/alpaka_NoBands_skymask_nside64.npy")
                nside = 64

                dec = []
                ra = []

                def check_mask(ra, dec, mask, nside):
                    idx = hp.ang2pix(nside, np.pi / 2. - dec, ra)
                    return mask[idx]

                for i in np.arange(n):
                    while(True):
                        rai = np.random.uniform(0., 2*np.pi)
                        deci = np.arcsin(np.random.uniform(-1., 1.))
                        cond = check_mask(rai, deci, alpaka_mask, nside)
                        if cond:
                            ra.append(rai)
                            dec.append(deci)
                            break

                dec = np.array(dec)
                ra = np.array(ra)

                v = core.ang2vec(ra, dec)
                mcat = self.mcat

            elif self.boots == 6:
                # Generate source positions within a max_shift from the
                # original position in the catalogue. Avoid galactic plane.
                # Preserve total number of sources.

                min_sindec = np.sin(self.dec_range[0])
                max_sindec = np.sin(self.dec_range[1])

                def gen_ra_dec(m):
                    if max_shift is not None:
                        dec, ra = core.psi_to_dec_and_ra(
                            self.random,
                            self.catalogue['dec'][m],
                            self.catalogue['ra'][m],
                            np.sqrt(self.random.uniform(
                                0, max_shift**2, np.count_nonzero(m)))
                        )
                    else:
                        ra = self.random.uniform(
                            0., 2.*np.pi, np.count_nonzero(m))
                        dec = np.arcsin(self.random.uniform(
                            min_sindec, max_sindec, np.count_nonzero(m))
                        )

                    return ra, dec

                def check_plane(ra, dec):
                    _, lat = core.galactic(ra, dec)

                    return np.fabs(lat) < self.mlat

                def check_hemisphere(dec):
                    return np.logical_or(
                        np.sin(dec) < min_sindec,
                        np.sin(dec) > max_sindec
                    )

                m = np.ones(n, dtype=bool)
                ra, dec = gen_ra_dec(m)
                m = np.logical_or(
                    check_plane(ra, dec),
                    check_hemisphere(dec)
                )
                while np.any(m):
                    ra[m], dec[m] = gen_ra_dec(m)
                    m = np.logical_or(
                        check_plane(ra, dec),
                        check_hemisphere(dec)
                    )

                v = core.ang2vec(ra, dec)
                mcat = self.mcat

            else:
                raise ValueError(
                    "Don't know scrambling {0:s}".format(self.boots))

        else:
            v = self.v
            mcat = self.mcat

        return v, mcat

    @core.timeit
    def trials(self, n_trials, n_jobs=1, max_shift=None):
        '''Perform trials on randomized data

        Parameters
        -----------
        n_trials : int
            Number of trials to use

        Returns
        ----------
        np.vstack(arr) : stacked array
        '''

        args = [(self, True, max_shift, self.random.randint(2**32))
                for i in np.arange(n_trials)]
        result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(seeder)(arg)
                                                    for arg in args)

        return np.vstack(result)


class BiasedCorrelator(Correlator):
    '''Correlator that biases some samples based on the correlation of prior
    samples.

    Parameters
    -----------
    catalogue : structured array
        Catalogue arrays with important fields: ra, dec, FOM.
    bins : array
        Bins to use for figure of merit.
    sigma : array
        Uncertainty of the Secondaries.

    Attributes
    -----------
    cosSig = ndarray
        Stores cosine of Secondaries uncertainty
    '''
    _w_scramble = False

    def __init__(self, catalogue, bins, sigma, seed=None, boots=None, **kwargs):

        super(BiasedCorrelator, self).__init__(
            catalogue, bins, seed, boots, **kwargs)

        self.cosSig = np.cos(sigma)

        return

    def add_secondary(self, name, sample):
        '''Add EventSample to Secondaries

        Parameters
        ----------
        name : string
            Name of the sample.
        sample : EventSample
            Event sample to add to Secondaries.
        '''

        assert(not (self.secondaries.has_key(name)
               or self.primaries.has_key(name)))

        assert(isinstance(sample, UHECRSample))

        self.secondaries[name] = sample

        return

    @ property
    def scramble(self):
        return self._v_scramble, self._u_scramble, self._w_scramble

    @ scramble.setter
    def scramble(self, value):
        '''Define the type of scrambling to apply.

        Parameters
        ----------
        value : int
            value 0  : Scramble only sources coordinates.
            value 1  : Scramble only primaries coordinates.
            value -1 : Scramble sources, primaries and secondaries coordinates.
            value 2  : Scramble only secondaries coordinates.
        '''
        value = int(value)
        if value == 0:
            print("Set scrambling to catalogue")
            self._v_scramble = True
            self._u_scramble = False
            self._w_scramble = False
        elif value == 1:
            print("Set scrambling to primaries")
            self._v_scramble = False
            self._u_scramble = True
            self._w_scramble = False
        elif value == 2:
            print("Set scrambling to secondaries")
            self._v_scramble = False
            self._u_scramble = False
            self._w_scramble = True
        elif value == -1:
            print("Scramble everything")
            self._v_scramble = True
            self._u_scramble = True
            self._w_scramble = True
        else:
            raise ValueError("Scrambling mode none of (0, 1, 2)")

        return

    @ Correlator.seed.setter
    def seed(self, value):
        val = int(value)

        Correlator.seed.fset(self, val)

        for sec in self.secondaries.itervalues():
            sec.random.seed(self.random.randint(2**32))

        return

    def __call__(self, scramble=False):
        '''Calculate the number of counterparts between sources selected with
        the primaries and the secondaries.

        Parameters
        ----------
        scramble : bool
            Set the scrambling to apply (default: False)

        Returns
        ----------
        P_cp : ndarray
            Number of primaries with a counterpart per bin.
        S_cp : ndarray
            Number of secondaries with a counterpart that has at least one
            primary counterpart, per bin.
        Src_cp : mask
            Sources that are counterparts for both a primary and a seconday.
        '''
        v_scramble, u_scramble, w_scramble = self.scramble if scramble else (
            False, False, False)

        v, mcat = self.get_v(scramble=v_scramble)

        # Get the Number of primaries with one counterpart (P_cp) and the mask
        # with sources that are counterparts for at least one primary (m_cp).
        P_cp, m_cp = super(BiasedCorrelator, self)._get_cp(v, mcat, u_scramble)

        if len(self.secondaries) < 1:
            return P_cp, None, None

        # secondary samples
        S_cp, Src_cp = self._get_cp(v, mcat, m_cp, w_scramble)

        return P_cp, S_cp, Src_cp

    def _get_cp(self, v, mcat, m_cp, scramble):
        '''Calculates the number of counterparts between v and secondaries.

        Parameters
        ----------
        v : array
            Array containing the 3D-vector positions of the Sources.
        mcat : ndarray
            Mask that stores the information about which bin contains which sources.
        m_cp : ndarray
            Mask that stores the sources that are counterparts for at least one
            primay.
        scramble : bool
            Scrambling the primaries

        Returns
        ----------
        N_cp : ndarray
            Number of events with one counterpart.
            * N_cp[0] : Number of secondaries with a counterpart selected for
                        having a primary.
            * N_cp[1] : Number of secondaries with a counterpart selected for
                        NOT having a primary.
            * N_cp[2] : Number of secondaries with a counterpart.
        M_cp : mask
            Mask with the sources that are counterparts of at least one
            neutrino and at least one CR.
        '''
        assert(mcat.shape == m_cp.shape)

        cS = self.cosSig

        # Calculate three counterparts: correlating, non-correlating, all
        N_cp = np.empty((3, len(cS), len(mcat)), dtype=[
            (k, np.uint16) for k in self.secondaries.iterkeys()])

        # sources that have CR counterparts
        m_cr = np.zeros(cS.shape + m_cp.shape, dtype=np.bool)

        for key, sam in self.secondaries.iteritems():
            w = sam.get_vector(scramble=scramble)

            # calculate angular distances between secondaries and sources
            cosD = core.dist(w, v)

            # get counterpart sources within uncertainty sigma, append dummy axis for combination
            msigma = (cosD >= cS[:, np.newaxis, np.newaxis])[:, np.newaxis]

            # neutrino correlation
            M = np.empty((len(cS), len(mcat)) + cosD.shape, dtype=np.bool)
            np.logical_and(msigma, m_cp[np.newaxis, :, np.newaxis], M)
            N_cp[key][0] = M.any(axis=-1).sum(axis=-1)
            m_cr |= M.any(axis=-2)

            # no neutrino correlation
            np.logical_and(msigma, (mcat & (~m_cp))[
                           np.newaxis, :, np.newaxis], M)
            N_cp[key][1] = M.any(axis=-1).sum(axis=-1)

            # only Cosmic Rays
            np.logical_and(msigma, mcat[np.newaxis, :, np.newaxis], M)
            N_cp[key][2] = M.any(axis=-1).sum(axis=-1)

        M_cp = np.empty(N_cp.shape[1:], dtype=[("CR", np.uint16)])
        M_cp["CR"] = m_cr.sum(axis=-1)

        return N_cp, M_cp

    def get_counterparts(self, binval, sigma, verbose=True, invert=False, graph=None):
        '''Return mask of counterparts for events. Not on scrambled data.
        Added sigma for cosmic ray data

        '''
        result, mcat = super(BiasedCorrelator, self).get_counterparts(
            binval, verbose, invert=invert, graph=graph)

        n_sec_cp = np.zeros(len(self.catalogue), dtype=np.int)

        for key, sam in self.secondaries.iteritems():
            w = sam.get_vector()

            cosD = core.dist(w, self.v)

            m = cosD >= np.cos(sigma)

            result[key] = np.any(m & mcat, axis=1)

            n_sec_cp += np.sum(m & mcat, axis=0)

            if verbose:
                print("{0:10s}: {1:d} biased counterparts above {2:.2f}".format(
                    key, np.count_nonzero(result[key]), binval))

                if np.count_nonzero(result[key]) < 1:
                    continue

                print("\t{0:10s} - {1:22s} - {2:5s}".format("Event",
                      "Source", "Offset (Fraction)"))
                for m_i, d_i, ev_i in zip(m & mcat, cosD, sam.events):
                    if np.all(~m_i):
                        continue
                    idx = self.catalogue[m_i]["id"]
                    D = np.degrees(np.arccos(d_i[m_i]))
                    F = (1. - d_i[m_i]) / (1. - np.cos(sigma))

                    if graph is not None:
                        if not hasattr(graph, "nodesize"):
                            graph.nodesize = {}

                        graph.add_node("{0:s} {1:s}".format(key, ev_i["id"]))
                        # graph.nodesize["{0:s} {1:d}".format(key, int(ev_i["id"]))] = ev_i["E"]**2

                    s = np.argsort(idx)
                    for id, D, f in zip(idx[s], D[s], F[s]):
                        # print("\t{0:6f} - {1:24s} - {2:5.2f} ({3:7.2%})".format(ev_i["id"], id, D, 1-f))

                        if graph is not None:
                            graph.add_node(id)
                            graph.add_edge("{0:s} {1:s}".format(key, ev_i["id"]), id,
                                           weight=1-f, distance=D)

        if verbose:
            print("Number of sec. counterparts per source: {0:d}".format(
                np.sum(n_sec_cp > 0)))
            s = np.argsort(n_sec_cp)[::-1]
            n_sec_cp = n_sec_cp[s]
            for cat_i, ni in zip(self.catalogue[s][mcat[s]], n_sec_cp[mcat[s]]):
                print(
                    "\t{0:22s} - {1:2d} sec. counterpart(s)".format(cat_i["id"], ni))

        if invert and graph is not None:
            for cat in self.catalogue[mcat]:
                if cat["id"] not in graph:
                    graph.add_node(cat["id"])

        return result, mcat

    @ core.timeit
    def trials(self, n_trials, n_jobs=1, **kwargs):
        '''Perform trials on randomized data

        Parameters
        -----------
        n_trials : int
            Number of trials to use

        Returns
        ----------
        Stacked array with shape (n_trials, 3).

        '''
        args = [(self, True, self.random.randint(2**32))
                for i in np.arange(n_trials)]
        result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(seeder)(arg)
                                                    for arg in args)

        # stack output, new axis for secondaries that have multiple cases of correlation
        return (np.vstack([r[0] for r in result]),
                np.vstack([r[1][np.newaxis] for r in result]),
                np.vstack([r[2][np.newaxis] for r in result]))


class EventSample(object):
    '''Class handling events.

    Attributes
    -----------
    events : ndarray
        Contains coordinates of events ("ra", "dec") and angular
        uncertainty ("sigma").
    u : 3D ndarray
        Contains the position of the events as 3D vector.
    N : int
        Number of events.
    seed : int
        Seed.
    random : numpy.random.RandomState
        Random number generator.
    '''

    def __init__(self, events, seed=None):
        '''
        Parameters
        ----------
        events : ndarray
            Contains coordinates of events ("ra", "dec").
            May contain the angular uncertainty "sigma",
            if not it has to be provide as argument.
        seed : int | None
            Seed (default: None)
        '''

        self.events = events

        self.u = core.ang2vec(self.events["ra"], self.events["dec"])

        self.N = len(self.events)

        if seed is None:
            # print("{0:s} - Initialize with random seed".format(self.__repr__()))
            self.random = np.random.RandomState()
        else:
            # print("{0:s} - Initialize with seed {1:d}".format(self.__repr__(), seed))
            self.random = np.random.RandomState(seed)

        return

    def get_vector(self, scramble=False):
        '''Return 3D vector arrays for events and sigma. Scramble if wanted.

        Parameters
        ----------
        scramble : bool
            Scramble events coordinates (default: False)

        Returns
        ----------
        u : 3D ndarray
            Array with events coordinates as 3D vectors
        sigma : array
            Array with the angular uncertainty of the events coordinates.
        '''
        if scramble:
            n = self.N

            idx = np.arange(n)

            u = self.u[idx]

            # ra, dec = core.vec2ang(u)
            # print("I'm scrambling")
            # ra = self.random.uniform(0., 2*np.pi, n)

            # u = core.ang2vec(ra, dec)

            # scramble uniformly in azimuth
            sin = np.sqrt(1. - u[:, 2]**2)

            phi = self.random.uniform(0., 2. * np.pi, n)
            u[:, 0] = np.cos(phi)
            u[:, 1] = np.sin(phi)
            u[:, :2] *= sin[:, np.newaxis]

            # permute sigma values
            sigma = self.events["sigma"][self.random.permutation(idx)]

        else:
            u = self.u
            sigma = self.events["sigma"]

        return u, sigma

    def add_sigma_field(self, sigma):
        """ Sigma must be given in radians.
        """
        # Add the field 'sigma' if missing.
        if not hasattr(sigma, '__iter__'):
            sigma = np.ones_like(
                self.events['ra']) * float(sigma)

        else:
            if len(sigma) != len(self.events):
                raise RuntimeError(
                    "The code needs either one `sigma` value that is copied for "
                    "all the events, or as many `sigma` values as there are "
                    "events in the primary sample.")

        if 'sigma' not in self.events.dtype.names:
            new_dtype = np.dtype(
                self.events.dtype.descr + [('sigma', float)])
            events_addsigma = np.empty(self.events.shape, dtype=new_dtype)
            events_addsigma['ra'] = self.events['ra']
            events_addsigma['dec'] = self.events['dec']
            events_addsigma['logpVal'] = self.events['logpVal']
            events_addsigma['sigma'] = sigma

            self.events = events_addsigma

        else:
            print(
                "Changing `sigma` of the event sample..."
            )
            self.events['sigma'] = sigma

        return


class UHECRSample(EventSample):
    '''Class handling events like UHECR samples. Inherits from the EventSample
    class.

    Attributes
    ----------
    poisson :

    exposure :

    (TODO: Sigma handling)
    '''

    _poisson = True

    def __init__(self, events, exposure, seed=None, ngrid=1000, poisson=True):
        '''
        Parameters
        ----------
        events : ndarray
            Contains coordinates of Cosmic Ray events ("ra", "dec").
        exposure : spline
            Contains exposure of the CR Sample used.
        seed : int
            Seed (default: None)
        ngrid : int
            (default: 1000)
        poisson : bool
            Poissonian sampling (default: True).
        '''
        super(UHECRSample, self).__init__(events, seed=seed)

        self.poisson = poisson
        self._exposure = exposure
        x = np.linspace(self._exposure.get_knots().min(),
                        self._exposure.get_knots().max(),
                        ngrid + 1)
        self._xx = (x[1:] + x[:-1]) / 2.
        self._width = np.mean(np.diff(self._xx))
        self._yy = self._exposure(self._xx)
        self._yy /= self._yy.sum()

        return

    def _sampling(self):
        N = len(self.events)
        if self.poisson:
            N = self.random.poisson(N)

        x = self.random.choice(self.xx, N, p=self.yy)

        # smear uniformly
        x += self.width * self.random.uniform(-1., 1., size=N) / 2.

        return x

    @ property
    def poisson(self):
        return self._poisson

    @ poisson.setter
    def poisson(self, value):
        self._poisson = bool(value)

        return

    def get_vector(self, scramble=False):
        '''Return 3D vector arrays for events and sigma. Scramble if wanted.

        Parameters
        ----------
        scramble : bool
            Scramble events coordinates (default: False)

        Returns
        ----------
        u : 3D ndarray
            Array with events coordinates as 3D vectors
        sigma : array
            Array with the angular uncertainty of the events coordinates.
        '''
        if scramble:
            dec = self.sampling()

            N = len(dec)

            ra = self.random.uniform(0., 2.*np.pi, N)

            u = core.ang2vec(ra, dec)

        else:
            u = self.u

        return u


def seeder(args):
    '''
    Change the seed of the correlator when multiprocessing is used.
    '''
    corr, scramble, max_shift, seed = args
    if scramble:
        corr.seed = seed

    return corr(scramble=scramble, max_shift=max_shift)
