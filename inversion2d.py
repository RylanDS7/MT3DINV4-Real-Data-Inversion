# code by Rylan Stutters - github.com/RylanDS7

from simpeg import maps, utils, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
import numpy as np
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

for key in mtd.keys():
    rx_locs += [utm.from_latlon(mtd[key].latitude, mtd[key].longitude)[:2] + (mtd[key].elevation,)]
rx_locs = np.array(rx_locs)

# only use freqs that each reciever has data for
freqs2use = []
for f in mtd.get_periods()**-1:
    freq_count = 0
    for key in mtd.keys():
        if f in mtd[key].Z.frequency:
            freq_count += 1
        if freq_count == mtd.n_stations:
            freqs2use.append(f)


# ==================================================
# Setup mesh
# ==================================================

print("Building mesh")

ncx = 250 # number of core mesh cells
dx = 20 # base cell width
npad_x = 50  # number of padding cells
exp_x = 1.1 # expansion rate of padding cells
ncy = 150     
dy = 20
npad_y = 25
exp_y = 1.1  

hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
hy = [(dy, npad_y, -exp_y), (dy, ncy), (dy, npad_y, exp_y)]
hx_cells = discretize.utils.unpack_widths(hx)
hy_cells = discretize.utils.unpack_widths(hy)

x_center = rx_locs[:, 0].mean()
y_surface = rx_locs[:, 2].mean()          

x0 = x_center - hx_cells.sum() / 2
y0 = y_surface - ((dy * ncy) / 3) - (hy_cells.sum() / 2)
mesh = discretize.TensorMesh([hx, hy], origin=[x0, y0])

print(f"Mesh has {mesh.n_cells} cells")
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# mesh.plot_grid(ax=ax)
# ax.scatter(rx_locs[:,0], rx_locs[:, 2], color='orange', s=100, zorder=5)
# plt.show()
