# -*-coding:utf8-*-

# Python
from itertools import cycle

# SciPy
import healpy as hp
import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy.stats import percentileofscore, norm
from scipy.interpolate import InterpolatedUnivariateSpline

# matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
from matplotlib.colorbar import cm
import matplotlib.ticker
from matplotlib import cycler

# local
import lib.core as core

columnwidth = 3. + 3. / 8.
figsize = np.array([columnwidth, 3. / 4. * columnwidth])
GRsize = np.array([columnwidth, 2 * columnwidth / (1. + np.sqrt(5.))])

flux = (r"$F_\gamma\left(>50\,\mathrm{GeV}\right)"
        r"\:[\mathrm{ph}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]$")
lac_flux = flux.replace("GeV", "MeV").replace("50", "100")
threefhl_flux = flux.replace("50", "10")


def print_current_settings(settings: dict):
    r"""To print the selected settings at the beginning of a script."""

    print("\nCurrent settings:")
    for key, value in settings.items():
        print(key, " : ", value)
    print()


def matPars():
    matplotlib.rcParams.update({
        "figure.figsize": figsize,
        "figure.autolayout": True,
        "font.family": "sans-serif",
        "font.serif": ["Palatino"],
        "font.size": 10,
        "axes.prop_cycle": cycler("color", ["#d7191c", "#2b83ba", "#fdae61",
                                            "#756bb1"]),  # + cycler("marker", [".", 6, 7, 4]),
    })

    return


def bw():
    matplotlib.rcParams.update({"axes.prop_cycle": cycler(
        "color", [u"0.00", u"0.40", u"0.60", u"0.70"])})

    return


def gen_random_events(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    events = np.empty((N, ), dtype=[("ra", np.float),
                                    ("dec", np.float),
                                    ("sigma", np.float),
                                    ("id", np.float)])

    events["ra"] = np.random.uniform(0., 2.*np.pi, N)
    events["dec"] = np.arcsin(np.random.uniform(-1., 1., N))
    events["sigma"] = np.random.lognormal(mean=np.log(np.radians(5.)),
                                          sigma=0.25, size=N)
    events["id"] = np.arange(N)

    return events


def gen_random_exposed(N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def f(x): return np.exp(np.cos(xx)/np.radians(25.)**2)

    xx = np.linspace(-np.pi/2., np.pi/2., 100 + 2)[1:-1]
    spl = InterpolatedUnivariateSpline(xx, f(xx), k=1)

    events = np.empty((N, ), dtype=[("ra", np.float),
                                    ("dec", np.float),
                                    ("E", np.float),
                                    ("id", np.float)])

    events["dec"] = core.sampling(spl, N, 1000, poisson=False).next()
    events["ra"] = np.random.uniform(0., 2.*np.pi, N)
    events["E"] = np.random.pareto(1, N)
    events["id"] = np.arange(N)

    return events, spl


def gen_random_cat(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    from lib.core import inv_galactic
    gal_w = np.radians(15.)

    cat = np.empty((N, ), dtype=[("ra", np.float),
                                 ("dec", np.float),
                                 ("FOM", np.float),
                                 ("id", "S10")])

    cat["ra"] = np.random.uniform(0., 2.*np.pi, N)
    cat["dec"] = np.random.uniform(-1. + np.sin(gal_w), 1. - np.sin(gal_w), N)
    cat["dec"] += np.sign(cat["dec"]) * np.sin(gal_w)
    cat["dec"] = np.arcsin(cat["dec"])
    cat["ra"], cat["dec"] = inv_galactic(cat["ra"], cat["dec"])
    cat["FOM"] = np.random.exponential(size=N)
    cat["id"] = np.arange(N).astype("S10")

    return cat


def plot_events(ax, events, diff_tracks=True, errors=True, **kwargs):
    color = kwargs.pop("color", "black")
    linewidth = kwargs.pop("linewidth", 1)
    kwargs.setdefault("markersize", 5)
    kwargs.setdefault("markeredgewidth", 1)

    for ev_i in events:
        if diff_tracks:
            sigma_cond = ev_i["sigma"] > np.radians(5.)
        else:
            sigma_cond = True

        if "sigma" in ev_i.dtype.names and sigma_cond:
            n = 1000
            x1 = np.zeros(n)
            y1 = np.full(n, np.pi/2.)
            x2 = np.full(n, ev_i["ra"])
            y2 = np.full(n, ev_i["dec"])
            x3 = np.linspace(0., 2.*np.pi, n + 1)[:-1]
            y3 = np.pi/2. - np.full(n, ev_i["sigma"])

            x, y = rotate(x1, y1, x2, y2, x3, y3)

            x, y = skymap_data(x, y)

            ds = np.sqrt((np.roll(y, 1) - y)**2 + np.cos(y)
                         ** 2 * (np.roll(x, 1) - x)**2)
            m = np.where(ds > 2. * np.radians(360.) / n)[0]

            xx = list()
            yy = list()
            if len(m) == 1:
                x = np.roll(x, -m[0])
                y = np.roll(y, -m[0])
                xx.append(x)
                yy.append(y)
            elif len(m) >= 2:
                xx.append(x[0:m[0]])
                yy.append(y[0:m[0]])
                for i in np.arange(len(m) - 1):
                    xx.append(x[m[i]:m[i+1]])
                    yy.append(y[m[i]:m[i+1]])
                xx.append(x[m[-1]:])
                yy.append(y[m[-1]:])
            else:
                xx.append(x)
                yy.append(y)

            if errors:
                for x, y in zip(xx, yy):
                    ax.plot(x, y, color=color, linewidth=linewidth)

            marker = "o"
        else:
            marker = "x"

        ax.plot(*skymap_data(ev_i["ra"], ev_i["dec"]), color=color,
                marker=marker, **kwargs)

    return


def plot_cat(ax, cat, scale=1., **kwargs):
    cat = np.copy(cat)

    kwargs.setdefault("marker", "o")
    kwargs.setdefault("edgecolor", "none")
    kwargs.setdefault("color", ax._get_lines.prop_cycler.next()["color"])

    scores = np.concatenate([[0], cat["FOM"], [1.2 * cat["FOM"].max()]])

    s = kwargs.pop("s", -np.log10([1. - percentileofscore(scores, s) / 100.
                                   for s in cat["FOM"]]))

    if scale > 0.:
        f = scale * 10. * np.sqrt(2000. / len(cat)) * s
    else:
        f = np.abs(scale)

    ax.scatter(*skymap_data(cat["ra"], cat["dec"]), s=f, **kwargs)

    return


def skymap_data(ra, dec):
    # ra goes from 0 to 2pi
    # 0 should be mapped to +pi, 2pi to -pi
    ra = np.pi - ra

    return ra, dec


def skymap_hphist(plt, fig, ax, ev, nside=128, N=2500, sep=0., **kwargs):

    ret = np.zeros(hp.nside2npix(nside), dtype=np.float)
    theta, ra = hp.pix2ang(nside, np.arange(len(ret)))
    dec = np.pi / 2. - theta

    for Ev in ev:
        for ev_i in Ev:
            m = (np.cos(ra - ev_i["ra"]) * np.cos(dec) * np.cos(ev_i["dec"])
                 + np.sin(dec) * np.sin(ev_i["dec"]) > np.cos(ev_i["sigma"]))

            ret[m] += 1

    xx = np.linspace(np.pi, -np.pi, 2 * N)
    yy = np.linspace(np.pi, 0., N)
    X, Y = np.meshgrid(xx, yy)

    r = hp.rotator.Rotator(rot=(-180., 0., 0.), inv=True)

    YY, XX = r(Y.ravel(), X.ravel())

    pix = hp.ang2pix(nside, YY, XX)
    Z = np.reshape(ret[pix], X.shape)

    lon = np.linspace(-np.pi, np.pi, 2 * N)
    lat = np.linspace(-np.pi / 2., np.pi / 2., N)

    cmap = cycle(kwargs.pop("cmaps", ["magma", "viridis"]))

    sep = np.unique(np.append(sep, [-np.pi/2., np.pi/2.]))

    #gridspec = GridSpec(15, len(sep) - 1)

    p = list()
    c = list()
    for i, (l, h) in enumerate(zip(sep[:-1], sep[1:])):
        m = (lat > l) & (lat <= h)
        if np.any(m):
            p.append(ax.pcolormesh(lon, lat[m], Z[m],
                                   # , Z[m].max()),
                                   cmap=plt.cm.get_cmap(cmap.next()),
                                   vmin=0, vmax=Z[m].max(),
                                   rasterized=True, edgecolor="None", **kwargs))
        if h < np.pi / 2.:
            ax.plot([-np.pi, np.pi], [h, h], color="grey")

        #cax = fig.add_subplot(gridspec[-2, i])
        cax = fig.add_subplot(8, 2, 15 + i)
        cbar = plt.colorbar(mappable=p[-1], cax=cax, orientation="horizontal")
        #cbar.set_label("${0:.0f}^\circ<\delta<{1:.0f}^\circ$".format(np.degrees(l), np.degrees(h)))
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5, integer=True)
        cbar.ax.tick_params(labelsize="small")
        cbar.update_ticks()

        c.append(cbar)

    return p, c


def skymap_nu_hphist(plt, fig, ax, ev, nside=128, N=2500, **kwargs):

    ret = np.zeros(hp.nside2npix(nside), dtype=np.float)
    theta, ra = hp.pix2ang(nside, np.arange(len(ret)))
    dec = np.pi / 2. - theta

    for Ev in ev:
        for ev_i in Ev:
            m = (np.cos(ra - ev_i["ra"]) * np.cos(dec) * np.cos(ev_i["dec"])
                 + np.sin(dec) * np.sin(ev_i["dec"]) > np.cos(ev_i["sigma"]))

            ret[m] += 1

    xx = np.linspace(np.pi, -np.pi, 2 * N)
    yy = np.linspace(np.pi, 0., N)
    X, Y = np.meshgrid(xx, yy)

    r = hp.rotator.Rotator(rot=(-180., 0., 0.), inv=True)

    YY, XX = r(Y.ravel(), X.ravel())

    pix = hp.ang2pix(nside, YY, XX)
    Z = np.reshape(ret[pix], X.shape)

    lon = np.linspace(-np.pi, np.pi, 2 * N)
    lat = np.linspace(-np.pi / 2., np.pi / 2., N)

    cmap = kwargs.pop("cmap", "viridis")

    p = ax.pcolormesh(lon, lat, Z,
                      cmap=plt.cm.get_cmap(cmap),
                      vmin=0, vmax=Z.max(),
                      rasterized=True, edgecolor="None", **kwargs)

    cax = fig.add_subplot(8, 3, 23)
    cbar = plt.colorbar(mappable=p, cax=cax, orientation="horizontal")
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5, integer=True)
    cbar.ax.tick_params(labelsize="small")
    cbar.update_ticks()

    return p, cbar


def skymap_plot(plt, data, cp, sigma, sep=np.radians([25.]), sources=True, **kw):
    fig, ax = skymap_ax(plt)

    ev = list()
    for k in data.secondaries.itervalues():
        ev.append(append_fields(k.events, ["sigma"], [
                  np.full(len(k.events), sigma)]))

    skymap_hphist(plt, fig, ax, ev, sep=sep, **kw)

    if sources:
        plot_cat(ax, data.catalogue, **ax._get_lines.prop_cycler.next())

    for k, m in cp.iteritems():
        if k in data.primaries:
            ev = data.primaries[k].events

            if np.any(m):
                plot_events(ax, ev[m], color="LightGray")
            if np.any(~m):
                plot_events(ax, ev[~m], color="DimGray")

        elif k in data.secondaries:
            continue
        else:
            raise KeyError("Don't know the sample", ev)

    return fig, ax


def skymap_ax(plt, projection="aitoff", equatorial=True, fontsize='small',    **kwargs):
    kwargs.setdefault("figsize", GRsize)

    def y_string(y, pos): return r"$0^\circ$" if np.fabs(
        y) < 0.01 else r"${0:+.0f}^\circ$".format(y * 180./np.pi)

    fig, ax = plt.subplots(subplot_kw=dict(projection=projection), **kwargs)

    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
        np.radians(np.linspace(-180., 180., 9)[1:-1])))
    # ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(
        np.radians([-120., -60., 60., 120.])))
    ax.xaxis.set_ticklabels([])

    ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
        np.radians([-85., -60., -30., 30., 60., 85.])))
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(
        np.radians([-60., -30., 0., 30., 60.])))
    ax.set_yticklabels([r"$-60^{\circ}$", r"$-60^{\circ}$", "",
                       r"$+30^{\circ}$", r"$+60^{\circ}$"], fontsize=fontsize)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(y_string))

    if equatorial:
        ax.text(1.005, 0.4, r"0h", horizontalalignment="left", verticalalignment="center",
                transform=ax.transAxes, fontsize=fontsize)
        ax.text(-0.005, 0.4, r"24h", horizontalalignment="right", verticalalignment="center",
                transform=ax.transAxes, fontsize=fontsize)

        ax.set_longitude_grid_ends(89.)

        ax.text(0.8, 0.0, "Eq. (J2000)",
                transform=ax.transAxes, fontsize=fontsize)

    else:
        ax.text(1.005, 0.4, r"$-180^{\circ}$", horizontalalignment="left", verticalalignment="center",
                transform=ax.transAxes, fontsize=fontsize)
        ax.text(-0.005, 0.4, r"$180^{\circ}$", horizontalalignment="right", verticalalignment="center",
                transform=ax.transAxes, fontsize=fontsize)

        ax.set_longitude_grid_ends(89.)

        ax.text(0.8, 0.0, "Galactic", transform=ax.transAxes)

    return fig, ax


def plot_pV(ax, data, pVal, pVal_err=None, N_obs=None, N_trials=None,
            leg=True, leg_loc="lower right", xlabel="FOM", log=False, combined=True,
            tnumber=True):

    keys = pVal.dtype.names

    if keys == None:
        keys = np.arange(pVal.shape[0])

    if N_obs is not None:
        assert(N_obs.dtype.names == N_trials.dtype.names)
        assert(N_obs.shape == N_trials.shape[1:])

    pVmin = 1.

    if log:
        xx = np.power(10., data.bins)
    else:
        xx = data.bins

    for k in keys:
        if (not combined) and (k == 'combined'):
            continue

        p = ax.semilogy(xx, pVal[k], nonposy="clip", label=k,
                        marker="o", markeredgecolor="none",
                        markersize=3)

        pVmin = min(pVmin, pVal[k].min())

        if pVal_err is not None and k in pVal_err.dtype.names:
            ax.fill_between(xx, pVal_err[k][0], pVal_err[k][1],
                            color=p[0].get_color(), alpha=0.25)

        if N_obs is None or k not in N_obs.dtype.names:
            continue

        for b, pV, N, m in zip(xx, pVal[k], N_obs[k],
                               np.mean(N_trials[k], axis=0)):
            col = p[0].get_color()
            col = "black"
            if tnumber:
                ax.text(b, 1.02*pV, "{0:d}".format(N), horizontalalignment="center",
                        verticalalignment="bottom", color=col, fontsize=8)
                ax.text(b, pV/1.02, "{0:.1f}".format(m), horizontalalignment="center",
                        verticalalignment="top", color=col, fontsize=8)

    eps = 0.075
    r = data.bins[-1] - data.bins[0]
    xmin = data.bins[0] - eps * r
    xmax = data.bins[-1] + eps * r
    if log:
        xmin = 10**xmin
        xmax = 10**xmax
    ax.set_xlim(xmin, xmax)
    #ax.set_xlim(data.bins[0] - 0.05 * r, data.bins[-1] + 0.05 * r)
    ax.set_ylim(ymax=2.)

    if leg:
        ax.legend(loc=leg_loc)

    if log:
        ax.semilogx()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("chance probability")

    return pVmin


def plot_pV2dim(plt, data, pV):

    sigma = np.around(np.degrees(np.arccos(data.cosSig)), decimals=2)

    fig, axs = plt.subplots(nrows=len(pV.dtype.names), ncols=pV.shape[0],
                            figsize=(5 * pV.shape[0], 5*len(pV.dtype.names)),
                            squeeze=False)

    vmin = np.floor(np.log10(np.amin([pV[k] for k in pV.dtype.names])))

    for j, (k, ax) in enumerate(zip(pV.dtype.names, axs)):
        for i, ax_i in enumerate(ax):
            pV_i = pV[k][i]

            p = ax_i.pcolormesh(  # data.bins, sigma,
                pV_i, vmin=10**vmin, vmax=1.,
                norm=colors.LogNorm(), cmap=cm.magma_r)
            plt.colorbar(mappable=p, ax=ax_i)

            ax_i.plot(np.arange(pV_i.shape[1]) + 0.5, np.argmin(pV_i, axis=0) + 0.5,
                      color="white", marker="x")

            lab = ["correlating", "non-correlating", "unbiased"][i]
            if j < 1:
                ax_i.set_title(lab)

            ax_i.set_xlabel("FOM/Flux")
            ax_i.set_ylabel(((k + " ") if i < 1 else "") + "$\sigma/1^\circ$")

            ax_i.set_xticks(np.arange(len(data.bins)) + 0.5)
            ax_i.set_xticklabels(["{0:.2f}".format(b) for b in data.bins],
                                 fontsize="small", rotation="vertical")
            ax_i.set_yticks(np.arange(len(sigma)) + 0.5)
            ax_i.set_yticklabels(["{0:.2g}".format(d) for d in sigma],
                                 fontsize="small")

    return fig, axs


def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
    r"""Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2). The
       rotation is performed on (ra3, dec3).
    """
    def cross_matrix(x):
        r"""Calculate cross product matrix
            A[ij] = x_i * y_j - y_i * x_j
        """
        skv = np.roll(np.roll(np.diag(x.ravel()), 1, 1), -1, 0)
        return skv - skv.T

    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)
    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    assert(len(ra1) == len(dec1)
           == len(ra2) == len(dec2)
           == len(ra3) == len(dec3))

    alpha = np.arccos(np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2)
                      + np.sin(dec1) * np.sin(dec2))
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T
    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T
    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    nTn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    R = np.array([(1.-np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, nTn_i, nx_i in zip(alpha, nTn, nx)])
    vec = np.array([np.dot(R_i, vec_i.T) for R_i, vec_i in zip(R, vec3)])

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    ra += np.where(ra < 0., 2. * np.pi, 0.)

    return ra, dec
