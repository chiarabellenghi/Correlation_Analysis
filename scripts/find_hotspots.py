from lib.hotspot_search import skyscan_handler
import numpy as np
import healpy as hp
import os
import argparse

if (__name__ == '__main__'):
    p = argparse.ArgumentParser(
        description="Find the hotspots in a skymap and store them.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--skymap_path",
        type=str,
        required=True,
        help="Path to the skymap."
    )
    p.add_argument(
        "--odir",
        type=str,
        required=True,
        help="Path to the folder to store the hotspots file."
    )
    p.add_argument(
        "--nside",
        type=int,
        default=None,
        help="The Healpy nside parameter. If None, it will be retrieved "
        "from the number of pixels in the skymap."
    )
    p.add_argument(
        "--dec_range",
        default=None,
        nargs='+',
        help="Declination range to search the hotspots."
    )

    args = vars(p.parse_args())
    print("\nCurrent settings:")
    for key, value in args.items():
        print(key, " : ", value)
    print()

try:
    pVals = hp.read_map(args['skymap_path'])
except OSError:
    skymap = np.load(args['skymap_path'])
    if 'pvalue' in skymap.dtype.names:
        pVals = -np.log10(skymap['pvalue'])
    elif 'pVal' in skymap.dtype.names:
        pVals = skymap['pVal']
    else:
        raise ValueError("I don't find the pvalues in the skymap.")
except ValueError:
    raise TypeError("I think I can\'t read this skymap format... Sorry.")

nside = args["nside"]
if nside is None:
    try:
        nside = hp.npix2nside(pVals.size)
    except ValueError:
        raise ValueError(
            "The number of pixels in the skymap does not correspond to any "
            "valid nside. In this case nside must be given as argument!")

    # Initialize the skyscan handler
    skyscan = skyscan_handler(nside=nside)
    skyscan.pix_map = np.arange(hp.nside2npix(nside))
    skyscan.log10p_map = pVals
else:
    # Initialize the skyscan handler
    skyscan = skyscan_handler(args['skymap_path'], nside=nside)

skyscan.prepare_full_scan(dec_range=args['dec_range'])
hotspots = skyscan.get_hotspots(log10p_threshold=2.0, psi_min=1.0)

os.makedirs(args['odir'], exist_ok=True)
if np.min(args['dec_range']) >= -5.:
    hemisphere = 'north'
    ofile = os.path.join(args['odir'], f'hotspots_{hemisphere}.npy')
elif np.max(args['dec_range']) <= -5:
    hemisphere = 'south'
    ofile = os.path.join(args['odir'], f'hotspots_{hemisphere}.npy')
else:
    ofile = os.path.join(args['odir'], f'hotspots.npy')

np.save(ofile, hotspots)
