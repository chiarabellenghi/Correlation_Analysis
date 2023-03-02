# -*-coding:utf8-*-

# Python
import os

# Scipy
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline

# local
from .correlation import BiasedCorrelator, EventSample, UHECRSample

dpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

# full degree binning
sigma_bins = np.radians(np.arange(0., 30. + 1., 1.)[1:])


def hours2dec(d, m, s, name):
    # check if the sources are positive or negative in declination
    hem = np.core.defchararray.find(name, "+") > 0
    return np.radians(d + np.where(hem, 1., -1.) * (m + s / 60.) / 60.)


def hours2ra(h, m, s):
    return np.pi / 12. * (h + (m + s / 60.) / 60.)


def load_neutrinos(dpath=dpath):

    print("Get neutrinos")

    arr = np.genfromtxt(os.path.join(dpath, "IceCube_deg_all_E_gt_60_TeV_real_and_tracks_err_lt_20.dat"),
                        usecols=[0, 1, 2, 3], names=["id", "dec", "ra", "sigma"])
    print("\t{0:d} events".format(len(arr)))

    for i in arr.dtype.names:
        if i == "id":
            continue
        arr[i] = np.radians(arr[i])

    return arr


def load_HESE(full=True, alerts=False, dpath=dpath):
    print("Get HESE neutrinos")

    if full:
        arr = np.genfromtxt(os.path.join(dpath, "IceCube_HESE_6yr.data"),
                            usecols=[0, 1, 3, 4, 5, 6],
                            dtype=[('id', 'S6'), ('E', np.float32),
                                   ('dec', np.float32), ('ra', np.float32),
                                   ('sigma', np.float32), ('Topology', 'S6')],
                            missing_values="---")
    else:
        arr = np.genfromtxt(os.path.join(dpath, "IceCube_HESE_4yr.data"),
                            usecols=[0, 1, 3, 4, 5, 6],
                            dtype=[('id', 'S6'), ('E', np.float32),
                                   ('dec', np.float32), ('ra', np.float32),
                                   ('sigma', np.float32), ('Topology', 'S6')],
                            missing_values="---")

    if alerts:
        arr2 = np.genfromtxt(os.path.join(dpath, "IceCube_HESE_Alerts.dat"),
                             usecols=[0, 1, 3, 4, 5, 6],
                             dtype=[('id', 'S6'), ('E', np.float32),
                                    ('dec', np.float32), ('ra', np.float32),
                                    ('sigma', np.float32), ('Topology', 'S6')],
                             missing_values="---")
        arr = np.concatenate((arr, arr2))

    print("\t{0:d} events".format(len(arr)))
    m = np.in1d(arr["Topology"], ["Track", "Shower"])
    print("\t\tRemove {0:d} background events".format((~m).sum()))
    arr = arr[m]

    for i in ["ra", "dec", "sigma"]:
        arr[i] *= np.pi / 180.

    return arr


def load_EHE(dpath=dpath):
    print("Get EHE neutrinos")

    arr = np.genfromtxt(os.path.join(dpath, "IceCube_EHE_Alerts.dat"),
                        usecols=[0, 1, 3, 4, 5, 6],
                        dtype=[('id', 'S6'), ('E', np.float32),
                               ('dec', np.float32), ('ra', np.float32),
                               ('sigma', np.float32), ('Topology', 'S6')],
                        missing_values="---")

    print("\t{0:d} events".format(len(arr)))

    for i in ["ra", "dec", "sigma"]:
        arr[i] *= np.pi / 180.

    return arr


def load_diffuse(dpath=dpath, full=True):
    print("Get diffuse neutrinos")

    if full:
        arr = np.genfromtxt(os.path.join(dpath, "IceCube_deg_8year_track.dat"),
                            usecols=[0, 4, 1, 2, 3, 5],
                            dtype=[('id', 'S6'), ('E', np.float32),
                                   ('dec', np.float32), ('ra', np.float32),
                                   ('sigma', np.float32), ('Topology', 'S6')],
                            missing_values="---")
    else:
        arr = np.genfromtxt(os.path.join(dpath, "IceCube_deg_6year_track.dat"),
                            usecols=[0, 4, 1, 2, 3, 5],
                            dtype=[('id', 'S6'), ('E', np.float32),
                                   ('dec', np.float32), ('ra', np.float32),
                                   ('sigma', np.float32), ('Topology', 'S6')],
                            missing_values="---")

    for i in ["ra", "dec", "sigma"]:
        arr[i] *= np.pi / 180.

    print("\t{0:d} events (1 already in HESE)".format(len(arr)))

    return arr


def load_hotspots(dpath=dpath, hemisphere=None):
    if hemisphere is None:
        return np.load(os.path.join(dpath, "hotspots.npy"))
    elif hemisphere in ['north', 'south']:
        return np.load(os.path.join(dpath, f"hotspots_{hemisphere}.npy"))
    else:
        raise ValueError(
            f"hemisphere = {hemisphere} is not supported. "
            "Please use one among {'north', 'south', None}."
        )


def load_TA(dpath=dpath, full=False):

    print("Get TA UHECR")

    arr = np.genfromtxt(os.path.join(dpath, "TAevents.txt"),
                        usecols=[0, 2, 3, 4], names=["id", "E", "ra", "dec"])

    if full:
        brr = np.genfromtxt(os.path.join(dpath, "TAevents_new.txt"),
                            usecols=[0, 2, 3, 4], names=["id", "E", "ra", "dec"])
        arr = np.concatenate((arr, brr))

    print("\t{0:d} events".format(len(arr)))

    for i in ["ra", "dec"]:
        arr[i] = np.radians(arr[i])

    return arr


def load_TA_exposure(dpath=dpath):

    arr = np.genfromtxt(os.path.join(dpath, "TAExposure.csv"), delimiter=",")

    arr[:, 0] = np.radians(arr[:, 0])

    return InterpolatedUnivariateSpline(arr[:, 0], arr[:, 1], k=1)


def load_Auger(full=True, low_e=False, dpath=dpath):

    print("Get Auger UHECR")

    if (full and not low_e):
        orr = np.genfromtxt(os.path.join(dpath, "auger_full_a8.dat"),
                            usecols=[0, 1, 2, 3, 5], names=["year", "day", "dec", "ra", "weight"])

        arr = np.empty(len(orr), dtype=[("id", "S8"), ("E", np.float16),
                                        ("ra", np.float16), ("dec", np.float16),
                                        ("weight", np.float16)])

        arr["id"] = "{0}_{1}".format(orr["year"], orr["day"])
        arr["E"] = np.zeros(len(orr))
        arr["weight"] = orr["weight"]
        for i in ["ra", "dec"]:
            arr[i] = np.radians(orr[i])

    elif (full and low_e):
        orr = np.genfromtxt(os.path.join(dpath, "auger_full_4-8.dat"),
                            usecols=[0, 1, 2, 3, 5], names=["year", "day", "dec", "ra", "weight"])

        arr = np.empty(len(orr), dtype=[("id", "S8"), ("E", np.float16),
                                        ("ra", np.float16), ("dec", np.float16),
                                        ("weight", np.float16)])

        arr["id"] = "{0}_{1}".format(orr["year"], orr["day"])
        arr["E"] = np.zeros(len(orr))
        arr["weight"] = orr["weight"]
        for i in ["ra", "dec"]:
            arr[i] = np.radians(orr[i])

    else:
        arr = np.genfromtxt(os.path.join(dpath, "AugerEvents-Ecut.txt"),
                            usecols=[0, 2, 3, 4], names=["id", "E", "ra", "dec"])
        for i in ["ra", "dec"]:
            arr[i] = np.radians(arr[i])

    print("\t{0:d} events".format(len(arr)))

    return arr


def load_Auger_exposure(dpath=dpath):

    arr = np.genfromtxt(os.path.join(
        dpath, "AugerExposure.csv"), delimiter=",")

    arr[:, 0] = np.radians(arr[:, 0])

    return InterpolatedUnivariateSpline(arr[:, 0], arr[:, 1], k=1)


def load_2whsp(dpath=dpath, name=None, suff=None):

    print("Load 2WHSP")

    if suff:
        raise ValueError("2WHSP does not use suffixes")

    if name is None:
        fname = "1_2_WHSP_bII_gt_10.dat"
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    arr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[2, 3, 4, 5, 6, 7, 12],
                        )

    nrr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=1, dtype="S25")

    print("\tFound {0:d} sources".format(len(arr)))

    orr = np.empty(len(arr), dtype=[("ra", np.float), ("dec", np.float),
                                    ("FOM", np.float), ("id", "S22")])

    orr["id"] = nrr
    orr["ra"] = hours2ra(arr[:, 0], arr[:, 1], arr[:, 2])
    orr["dec"] = hours2dec(arr[:, 3], arr[:, 4], arr[:, 5], orr["id"])
    orr["FOM"] = arr[:, 6]

    return orr


def load_3hsp(dpath=dpath, name=None, suff=None):

    print("Load 3HSP")

    if suff:
        raise ValueError("3HSP does not use suffixes")

    if name is None:
        fname = "3HSP.csv"
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    arr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[2, 3, 4],
                        delimiter=",")

    nrr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=1,
                        dtype="S22",
                        delimiter=",")

    print("\tFound {0:d} sources".format(len(arr)))

    orr = np.empty(len(arr), dtype=[("ra", np.float), ("dec", np.float),
                                    ("FOM", np.float), ("id", "S22")])

    orr["id"] = nrr
    orr["ra"] = np.radians(arr[:, 0])
    orr["dec"] = np.radians(arr[:, 1])
    orr["FOM"] = arr[:, 2]

    return orr


def load_2fhl(dpath=dpath, name=None, suff="HBL", bii=10):

    if suff == "all":
        suff = ""
    elif suff:
        suff = "." + suff

    if name is None:
        fname = "2FHL_bii_gt_{0:02d}{1:s}.dat".format(bii, suff)
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    arr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[4, 5, 6, 7, 8, 9, 10],
                        )

    print("\tFound {0:d} sources".format(len(arr)))

    orr = np.empty(len(arr), dtype=[("ra", np.float), ("dec", np.float),
                                    ("FOM", np.float), ("id", "S17")])

    nrr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[2], dtype="S17")

    orr["id"] = nrr
    orr["ra"] = hours2ra(arr[:, 0], arr[:, 1], arr[:, 2])
    orr["dec"] = hours2dec(arr[:, 3], arr[:, 4], arr[:, 5], orr["id"])
    orr["FOM"] = np.log10(arr[:, 6])

    return orr


def load_3lac(dpath=dpath, name=None, suff="HBL"):

    if name is None:
        fname = "Fermi_3LAC_{0:s}.dat".format(suff)
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    arr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[1, 2, 3, 4, 5, 6, 10],
                        )

    print("\tFound {0:d} sources".format(len(arr)))

    orr = np.empty(len(arr), dtype=[("ra", np.float), ("dec", np.float),
                                    ("FOM", np.float), ("id", "S17")])

    nrr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[8], dtype="S17")

    orr["id"] = nrr
    orr["ra"] = hours2ra(arr[:, 0], arr[:, 1], arr[:, 2])
    orr["dec"] = hours2dec(arr[:, 3], arr[:, 4], arr[:, 5], orr["id"])
    orr["FOM"] = np.log10(arr[:, 6])

    return orr


def load_2hwc(dpath=dpath, name=None):

    if name is None:
        fname = "2HWC.dat"
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    arr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[0, 2, 3, 4],
                        skip_header=1
                        )

    print("\tFound {0:d} sources".format(len(arr)))

    orr = np.empty(len(arr), dtype=[("ra", np.float), ("dec", np.float),
                                    ("FOM", np.float), ("id", "S17")])

    orr["id"] = arr[:, 0]
    orr["ra"] = np.radians(arr[:, 2])
    orr["dec"] = np.radians(arr[:, 3])
    orr["FOM"] = np.log2(arr[:, 1])

    return orr


def load_fl8y(dpath=dpath, name=None, suff="All"):

    if name is None:
        fname = "FL8Y_{0:s}.csv".format(suff)
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    arr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[1, 2, 5],
                        delimiter=",",
                        skip_header=1)

    print("\tFound {0:d} sources".format(len(arr)))

    orr = np.empty(len(arr), dtype=[("ra", np.float32), ("dec", np.float32),
                                    ("FOM", np.float32), ("id", "S17")])

    nrr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[0], dtype="S12",
                        delimiter=",",
                        skip_header=1)

    orr["id"] = [i.replace('FL8Y ', '') for i in nrr]
    orr["ra"] = np.radians(arr[:, 0])
    orr["dec"] = np.radians(arr[:, 1])
    orr["FOM"] = np.log10(arr[:, 2])

    return orr


def load_3fhl(dpath=dpath, name=None, suff="All"):

    if suff == "All":
        suff = ""
    elif suff:
        suff = "_" + suff

    if name is None:
        fname = "3FHL_V5.1_{1:s}.csv".format(suff)
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    arr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[1, 2, 3],
                        delimiter=",",
                        skip_header=1)

    print("\tFound {0:d} sources".format(len(arr)))

    orr = np.empty(len(arr), dtype=[("ra", np.float32), ("dec", np.float32),
                                    ("FOM", np.float32), ("id", "S17")])

    nrr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[0], dtype="S12",
                        delimiter=",",
                        skip_header=1)

    orr["id"] = [i.replace('3FHL ', '') for i in nrr]
    orr["ra"] = np.radians(arr[:, 0])
    orr["dec"] = np.radians(arr[:, 1])
    orr["FOM"] = np.log10(arr[:, 2])

    return orr


def load_alpaka(dpath=dpath, name=None, OIII=False):

    if name is None:
        fname = "ALPAKA_reduced.csv"
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    arr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[2, 3, 10, 16, 22],
                        delimiter=",",
                        skip_header=1,
                        filling_values="0")

    print("\tFound {0:d} sources".format(len(arr)))

    orr = np.empty(len(arr), dtype=[("ra", np.float), ("dec", np.float),
                                    ("FOM", np.float), ("id", "S23")])

    nrr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[0], dtype="S23",
                        delimiter=",",
                        skip_header=1)

    orr["id"] = nrr
    orr["ra"] = np.radians(arr[:, 0])
    orr["dec"] = np.radians(arr[:, 1])
    if OIII:
        orr["FOM"] = np.log10(arr[:, 2] + arr[:, 3])
    else:
        orr["FOM"] = arr[:, 4]

    return orr


def load_outflows(dpath=dpath, name=None):

    if name is None:
        fname = "Outflows.csv"
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    arr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[1, 2],
                        delimiter=",",
                        skip_header=1,
                        filling_values="0")

    print("\tFound {0:d} sources".format(len(arr)))

    orr = np.empty(len(arr), dtype=[("ra", np.float), ("dec", np.float),
                                    ("FOM", np.float), ("id", "S18")])

    nrr = np.genfromtxt(os.path.join(dpath, fname),
                        usecols=[0], dtype="S18",
                        delimiter=",",
                        skip_header=1)

    orr["id"] = nrr
    orr["ra"] = np.radians(arr[:, 0])
    orr["dec"] = np.radians(arr[:, 1])
    orr["FOM"] = np.zeros_like(arr[:, 0])

    return orr


def load_5bzcat(dpath=dpath, name=None):

    if name is None:
        fname = "5bzcat.csv"
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    df = pd.read_csv(
        os.path.join(dpath, fname), delimiter=",", usecols=[1, 2, 3, 4, 5],
        skiprows=1, header=None,
        names=["id", "classification", "ra", "dec", "FOM"],
        na_values='  ',
    )

    dtype = [
        ("id", "S18"),
        ("classification", "S21"),
        ("ra", float),
        ("dec", float),
        ("FOM", float),
    ]

    arr = df.to_numpy()
    orr = np.empty(len(df), dtype=dtype)

    for i, name in enumerate(df.columns):
        orr[name] = arr[:, i]

    # Where no redshift, set to 999
    orr['FOM'][np.isnan(orr['FOM'])] = 999

    # Cat Bl Lac Candidate
    m = orr["classification"] == b'BLLacCandidate'
    orr = orr[~m]

    orr['ra'] = np.radians(orr['ra'])
    orr['dec'] = np.radians(orr['dec'])

    print("\tFound {0:d} sources".format(len(orr)))

    return orr


def load_rfc(dpath=dpath, name=None):

    if name is None:
        fname = "RFC_subset.txt"
    else:
        fname = str(name)

    print("Load {0:s}".format(fname))

    df = pd.read_csv(
        os.path.join(dpath, fname), delimiter=",", usecols=[0, 1, 2, 5, 7],
        skiprows=1, header=None,
        names=["id", "ra", "dec", "FOM", "BZCat_name"],
        na_values='  ',
    )

    dtype = [
        ("id", "S18"),
        ("ra", float),
        ("dec", float),
        ("FOM", float),
        ("BZCat_name", "S21"),
    ]

    arr = df.to_numpy()
    orr = np.empty(len(df), dtype=dtype)

    for i, name in enumerate(df.columns):
        orr[name] = arr[:, i]

    orr['ra'] = np.radians(orr['ra'])
    orr['dec'] = np.radians(orr['dec'])

    print("\tFound {0:d} sources".format(len(orr)))

    return orr


catnames = ["2WHSP",
            "2FHL",
            "3LAC",
            "2HWC",
            "3FHL",
            "FL8Y",
            "ALPAKA",
            "OUTFLOWS",
            "3HSP",
            "5BZCat",
            "RFC"]

loaders = [load_2whsp,
           load_2fhl,
           load_3lac,
           load_2hwc,
           load_3fhl,
           load_fl8y,
           load_alpaka,
           load_outflows,
           load_3hsp,
           load_rfc]

catsuffs = [[""],  # 2whsp
            ["", "HBL", "non_HBL"],  # 2fhl
            ["all", "HBL", "FSRQ", "others"],  # 3lac
            [""],  # 2hwc
            ["All", "HBL", "HBLandUncl", "Blazar",
                "Non_HBL_Blazar", "Non_Blazar"],  # 3fhl
            ["All", "BLLacs", "FSRQ", "Blazar"],  # fl8y
            [""],  # alpaka
            [""],  # ouflows
            [""],  # 3hsp
            [""],  # 5bzcat
            [""],  # rfc
            ]

catbins = [[0., 0.316, 0.631, 1., 1.585, 1.995, 2.512, 3.162, 3.981],  # 2whsp
           #           [-11.13, -11., -10.9, -10.75, -10.6, -10.5, -10.45, -10.],
           [-11.12726117, -10.9473076, -10.8827287, -10.81815641, -10.75448733, -10.69250396, - \
               10.58703884, -10.48269263, -10.27818938, -9.97021222, -8.8827287],  # 2fhl
           [-10., -9.5, -9., -8.75, -8.5, -8.36, - \
               8.25, -8.15, -8., -7.5, -7.],  # 3lac
           [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
               11., 12., 13., 14., 15., 16.],  # 2hwc
           [-10.89075903, -10.61385789, -10.52709735, -10.45763526, -10.38544023, -10.298003, - \
               10.19155957, -10.0739989, -9.89736055,  -9.60469233,  -7.91542372],  # 3fhl
           [-13., -11.75, -11.5, -11.25, -11., -10.75, -10.5, -10, -9.],  # fl8y
           [[200, 500, 800, 1100, 1400, 1700, 2000., 4000], [38., 38.5, 39., 39.5,
                                                             40., 40.5, 41., 41.5, 42., 42.5, 43.]],  # Alpaka Avg.Vel, Alpaka OIII
           [0],  # Outflows
           [0., 1./32., 1./16, 1./8., 1./4., 1./2., 1., 2., 4.],  # 3hsp
           [0],   # 5BZCat
           [0]]   # RFC


def init_biascorr_newNeutrinos(cat, bins, seed=None, scramble=0, mlat=np.radians(10.), boots=0, TA_full=False, sigma_factor=1., **kwargs):

    data_path = kwargs.pop("dpath", dpath)
    cat = cat(dpath=data_path, **kwargs)

    if seed is None:
        seed = 1954, 1974, 1990, 2014, 1909, 1997, 3016
    elif type(seed) is int:
        seed = [seed * 1000 + i for i in range(7)]

    # HESE
    hese = load_HESE(dpath="../data", full=True, alerts=True)
    diffuse = load_diffuse(dpath="../data", full=True)
    ehe = load_EHE(dpath="../data")
    nu = np.concatenate((hese, diffuse, ehe))
    cut = (nu["E"] > 60.) & (nu["sigma"] < np.radians(20.))
    print("\tRemove {0:d} due to energy or reconstruction".format(
        (~cut).sum()))
    nu = nu[cut]
    m_tracks = nu["Topology"] == "Track"
    print("\t{0:d} Neutrinos of which {1:d} are tracks".format(
        len(nu), m_tracks.sum()))
    Cascades = EventSample(nu[~m_tracks], seed=seed[0])
    if np.not_equal(1., sigma_factor):
        print("Setting Tracks sigma to * {0}".format(sigma_factor))
        nu['sigma'] *= sigma_factor
    Tracks = EventSample(nu[m_tracks], seed=seed[1])

    Auger = UHECRSample(load_Auger(dpath=data_path, full=False),
                        load_Auger_exposure(dpath=data_path), seed=seed[4])
    TA = UHECRSample(load_TA(dpath=data_path, full=TA_full),
                     load_TA_exposure(dpath=data_path), seed=seed[5])

    C = BiasedCorrelator(
        cat, bins, sigma_bins, seed=seed[6], scramble=scramble, mlat=mlat, boots=boots)

    C.add_primary("Cascades", Cascades)
    C.add_primary("Tracks", Tracks)
    C.add_secondary("Auger", Auger)
    C.add_secondary("TA", TA)

    return C


def init_2whsp_newNeutrinos(**kwargs):
    return init_biascorr_newNeutrinos(loaders[0], catbins[0], **kwargs)


def init_2fhl_newNeutrinos(**kwargs):
    return init_biascorr_newNeutrinos(loaders[1], catbins[1], **kwargs)


def init_3lac_newNeutrinos(**kwargs):
    return init_biascorr_newNeutrinos(loaders[2], catbins[2], **kwargs)


def init_2hwc_newNeutrinos(**kwargs):
    return init_biascorr_newNeutrinos(loaders[3], catbins[3], **kwargs)


def init_3fhl_newNeutrinos(**kwargs):
    return init_biascorr_newNeutrinos(loaders[4], catbins[4], **kwargs)


def init_fl8y_newNeutrinos(**kwargs):
    return init_biascorr_newNeutrinos(loaders[5], catbins[5], **kwargs)


def init_alpaka_newNeutrinos(OIII=False, **kwargs):
    if OIII:
        catb = catbins[6][1]
    else:
        catb = catbins[6][0]
    return init_biascorr_newNeutrinos(loaders[6], catb, **kwargs)


def init_outflows_newNeutrinos(**kwargs):
    return init_biascorr_newNeutrinos(loaders[7], catbins[7], **kwargs)


def init_3hsp_newNeutrinos(**kwargs):
    return init_biascorr_newNeutrinos(loaders[8], catbins[8], **kwargs)
