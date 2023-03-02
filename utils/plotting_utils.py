# Stuff for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.size"]=18
mpl.rcParams["font.family"]="sans-serif"
mpl.rcParams["text.usetex"]=True
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')

ps_map_colors =  {'blue' : ((0.0, 0.0, 1.0),
                            (0.05, 1.0, 1.0),
                            (0.4, 1.0, 1.0),
                            (0.6, 1.0, 1.0),
                            (0.7, 0.2, 0.2),
                            (1.0, 0.0, 0.0)),
                  'green': ((0.0, 0.0, 1.0),
                            (0.05, 1.0, 1.0),
                            (0.5, 0.0416, 0.0416),
                            (0.6, 0.0, 0.0),
                            (0.8, 0.5, 0.5),
                            (1.0, 1.0, 1.0)),
                  'red':   ((0.0, 0.0, 1.0),
                            (0.05, 1.0, 1.0),
                            (0.5, 0.0416, 0.0416),
                            (0.6, 0.0416, 0.0416),
                            (0.7, 1.0, 1.0),
                            (1.0, 1.0, 1.0))}

ps_map = mpl.colors.LinearSegmentedColormap('ps_map', ps_map_colors, 256)

import healpy as hp
def plot_skymap(pVal_map):
    hp.projview(pVal_map, rot=(180,0,0), coord=["G"], graticule=True, 
                graticule_labels=True, unit="$-\log_{10}(p_{\mathrm{local}})$",
                cb_orientation="horizontal", projection_type='hammer', 
                min=0, max=6, flip='astro', rot_graticule=True,
                cmap=ps_map, longitude_grid_spacing=45, latitude_grid_spacing=30,
                override_rot_graticule_properties={'g_linestyle':":", 'g_alpha':1},
                fontsize={'cbar_label':20, 'cbar_tick_label':16, 'xtick_label':16, 'ytick_label':16},
                show_tickmarkers=True, cbar_ticks=[0,1,2,3,4,5,6],
                override_plot_properties={'cbar_label_pad': 5, 'cbar_pad': 0.07, "cbar_tick_direction": "in"},
                xtick_label_color="k", phi_convention="counterclockwise"
               )
    ax = plt.gca()
    ax.set_yticklabels([r"$-60^{\circ}$", r"$-30^{\circ}$", "$\\boldsymbol{24\mathrm{h}}$", r"$+30^{\circ}$", r"$+60^{\circ}$"], fontsize=16)
    ax.text(1.01, 0.5, "$\\boldsymbol{0\mathrm{h}}$", horizontalalignment="left", verticalalignment="center",
            transform=ax.transAxes, fontsize=16
           )
    return ax