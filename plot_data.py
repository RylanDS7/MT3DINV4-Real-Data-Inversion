# code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import discretize
import matplotlib.pyplot as plt

with np.load('out/inversion_results.npz') as data:
    minv_tetm = data['model'],
    data_model = data['dpred'],
    rho_est = data['rho_est'],
    freqs2use = data['freqs'],
    rx_locs = data['rx_locs']

#

ncx = 150 # number of core mesh cells
dx = 40 # base cell width
npad_x = 20  # number of padding cells
exp_x = 1.5 # expansion rate of padding cells
ncy = 75 # number of core mesh cells
dy = 40 # base cell width
npad_y = 20 # number of padding cells
exp_y = 1.5 # expansion rate of padding cells

hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
hy = [(dy, npad_y, -exp_y), (dy, ncy), (dy, npad_y, exp_y)]
hx_cells = discretize.utils.unpack_widths(hx)
hy_cells = discretize.utils.unpack_widths(hy)

x_center = rx_locs[:, 0].mean()
y_surface = rx_locs[:, 2].mean()          

x0 = x_center - hx_cells.sum() / 2
y0 = y_surface - (hy_cells.sum() / 2) - 100
mesh = discretize.TensorMesh([hx, hy], origin=[x0, y0])

#

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# conductivity
mtrue = np.log10(1/np.exp(rho_est))
clim = [np.log10(20), np.log10(10000)]

dat = mesh.plot_image(
    (mtrue),
    ax=ax,
    # grid=True,
    clim=clim,
    range_x=[rx_locs[:,0].min()-1000, rx_locs[:,0].max()+1000],
    range_y=[rx_locs[:,2].min()-2000, rx_locs[:,2].max()+100],
    pcolor_opts={"cmap": "Spectral"}
)

ax.set_title('Resistivity')
plt.colorbar(
    dat[0],
    cmap='Spectral', 
    label=r'Resistivity ($\Omega$m)', 
    ticks=[clim[0],clim[1]], 
    format="$10^{%.1f}$", 
    shrink=0.6
).ax.tick_params(labelsize=14)

plt.show()

