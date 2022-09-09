# -*-coding:utf8-*-

# Python
import time

# SciPy
import healpy as hp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import percentileofscore

# astropy coordinate transformations
from astropy.coordinates import SkyCoord
from astropy import units as u

comb = "combined"


def ang2vec(ra, dec):
    '''Turn angular vectors into 3-dim vectors in cartesian coordinates

    Parameters
    -----------
    ra : float, array-like, (N, )
        Rightascension values
    dec : float, array-like, (N, )
        Declination values

    Returns
    --------
    u : array-like, (N, 3)
        3dim vectors of angles

    '''
    ra = np.asarray(ra)
    dec = np.asarray(dec)

    # convert declination to zenith
    theta = np.pi/2. - dec

    return hp.ang2vec(theta, ra)


def vec2ang(vec):
    theta, phi = hp.vec2ang(vec)

    dec = np.pi/2. - theta

    return phi, dec


def dist(u, v):
    '''Calculate cosine angle using scipy.spatial.distance implementation

    Parameters
    -----------
    u, v : (N/M, 3) shape vectors
        Euclidean coordinates

    Returns
    --------
    D : array-like, (N, M)
        Distance metric in cos(Psi)

    '''

    # use cosine distance: D = 1 - u.v / |u||v| = 1 - cos(angle(u, v))
    D = cdist(u, v, "cosine")

    return 1. - D


def psi_to_dec_and_ra(rss, src_dec, src_ra, psi):
    """Generates random declinations and right-ascension coordinates for the
    given source location and opening angle `psi`.
    Parameters
    ----------
    rss : instance of np.random.mtrand.RandomState
        The instance of np.random.mtrand.RandomState to use
        for drawing random numbers.
    src_dec : float
        The declination of the source in radians.
    src_ra : float
        The right-ascension of the source in radians.
    psi : 1d ndarray of float
        The opening-angle values in radians.
    Returns
    -------
    dec : 1d ndarray of float
        The declination values.
    ra : 1d ndarray of float
        The right-ascension values.
    """
    if not isinstance(rss, np.random.mtrand.RandomState):
        raise TypeError("To be implemented.")

    psi = np.atleast_1d(psi)

    # Transform everything in radians and convert the source declination
    # to source zenith angle
    a = psi
    b = np.pi/2 - src_dec
    # b = src_dec
    c = src_ra

    # Random rotation angle for the 2D circle
    t = rss.uniform(0, 2*np.pi, size=len(psi))

    # Parametrize the circle
    x = (
        (np.sin(a)*np.cos(b)*np.cos(c)) * np.cos(t) +
        (np.sin(a)*np.sin(c)) * np.sin(t) -
        (np.cos(a)*np.sin(b)*np.cos(c))
    )
    y = (
        -(np.sin(a)*np.cos(b)*np.sin(c)) * np.cos(t) +
        (np.sin(a)*np.cos(c)) * np.sin(t) +
        (np.cos(a)*np.sin(b)*np.sin(c))
    )
    z = (
        (np.sin(a)*np.sin(b)) * np.cos(t) +
        (np.cos(a)*np.cos(b))
    )

    # Convert back to right-ascension and declination.
    # This is to distinguish between diametrically opposite directions.
    zen = np.arccos(z)
    azi = np.arctan2(y, x)

    dec = np.pi/2 - zen
    # dec = zen
    ra = np.pi - azi

    return (dec, ra)


def galactic(ra, dec):
    """ From equatorial to galactic coordinates.
    `ra` and `dec` must be given in radians.

    ra : float | array-like
        (List/array of) Right Ascension value(s) in radians.
    dec : float | array-like
        (List/array of) Declination value(s) in radians.

    Returns: galactic longitude and latitude.
    """
    c = SkyCoord(ra=ra*u.rad, dec=dec*u.rad, frame='icrs')
    gal = c.galactic

    return gal.l.to_value(unit=u.rad), gal.b.to_value(unit=u.rad)


def inv_galactic(lon, lat):
    """ From galactic to equatorial coordinates.
    `lon` and `lat` must be given in radians.

    lon : float | array-like
        (List/array of) galactic longitude value(s) in radians.
    dec : float | array-like
        (List/array of) galactic latitude value(s) in radians.

    Returns: Right Ascension and Declination.
    """
    c = SkyCoord(l=lon*u.rad, b=lat*u.rad, frame='galactic')
    eq = c.icrs

    return eq.ra.to_value(unit=u.rad), eq.dec.to_value(unit=u.rad)


def sampling(spl, N, npoints, poisson=True, random=np.random):
    '''Use a spline to sample N points using *npoints* gridpoints

    '''

    # create equidistant grid
    xx = np.linspace(spl.get_knots().min(), spl.get_knots().max(), npoints + 1)
    xx = (xx[1:] + xx[:-1]) / 2.
    width = np.mean(np.diff(xx))

    # create declination weights
    yy = spl(xx)
    yy /= yy.sum()

    while True:
        n = N if not poisson else random.poisson(N)

        x = random.choice(xx, n, p=yy)

        # smear x uniformly with grid size

        x += width * random.uniform(-1., 1., size=n) / 2.

        yield x


def timeit(method):
    '''Decorator that prints out timing statistics for use with other functions

    by Andreas Jung
    '''

    def timed(*args, **kw):
        '''Timing function passing the arguments and keywords to the method

        '''

        # first time stamp
        start = time.time()

        print("-- Calling function {0:s} of {1:s}".format(
            method.__name__, args[0].__repr__())
        )

        # call method
        result = method(*args, **kw)

        # second time stamp
        stop = time.time()

        sec = stop - start
        min, sec = divmod(sec, 60)
        hr, min = divmod(min, 60)
        day, hr = divmod(hr, 24)

        # give output
        print("--> Time ({0:s}, {1:s}): {2:2.0f}d {3:2.0f}:{4:2.0f}:{5:.2f}".format(
            args[0].__repr__(), method.__name__, day, hr, min, sec)
        )

        return result

    return timed
