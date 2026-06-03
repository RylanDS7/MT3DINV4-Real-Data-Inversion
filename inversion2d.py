# code by Rylan Stutters - github.com/RylanDS7

from simpeg import maps, utils, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
import numpy as np
import matplotlib
from pymatsolver import Pardiso
from simpeg.electromagnetics import natural_source as nsem
import matplotlib.pyplot as plt
import utm
import mtpy as mt


inversion_title = 'top_line_2d'

# ==================================================
# Load data
# ==================================================

mtc = mt.MTCollection()
mtc.open_collection(inversion_title)
mtd = mtc.to_mt_data()
mtc.close_collection()

# ==================================================
# Get locations and freqs
# ==================================================

rx_locs = []
elevation = []

for key in mtd.keys():
    rx_locs += [utm.from_latlon(mtd[key].latitude, mtd[key].longitude)[:2]]
    elevation += [mtd[key].elevation]

# only use freqs that each reciever has data for
freqs2use = []
for f in mtd.get_periods()**-1:
    freq_count = 0
    for key in mtd.keys():
        if f in mtd[key].Z.frequency:
            freq_count += 1
        if freq_count == mtd.n_stations:
            freqs2use.append(f)
