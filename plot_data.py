# code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import discretize
import matplotlib.pyplot as plt
import copy

inversion_title = 'P06cf'

with np.load(f'out/2dinversion_results_{inversion_title}.npz', allow_pickle=True) as data:
    minv_tetm = data['model']
    data_model = data['dpred']
    sigma_est = data['sigma_est']
    peris2use = data['peris']
    rx_locs2d = data['rx_locs2d']
    mesh = data['mesh'].item()

mesh = discretize.TensorMesh.deserialize(mesh)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# resistivity
mtrue = np.log10(1/np.exp(sigma_est))
clim = [np.log10(0.1), np.log10(1e5)]

cmap = copy.copy(plt.get_cmap("Spectral"))
cmap.set_over("white")

dat = mesh.plot_image(
    (mtrue),
    ax=ax,
    # grid=True,
    clim=clim,
    range_x=[rx_locs2d[:,0].min()-500, rx_locs2d[:,0].max()+500],
    range_y=[rx_locs2d[:,1].min()-1000, rx_locs2d[:,1].max()+100],
    pcolor_opts={"cmap": cmap}
)

plt.scatter(rx_locs2d[:,0], rx_locs2d[:,1], color='black', s=50, zorder=5)

ax.set_title('Resistivity')
cbar = plt.colorbar(
    dat[0],
    ticks=[clim[0],clim[1]], 
    format="$10^{%.1f}$", 
    shrink=0.6,
    extend='max'
)
cbar.ax.tick_params(labelsize=22)
cbar.set_label(r'Resistivity ($\Omega$m)', fontsize=24)

plt.title(f'2D Inversion Results - {inversion_title}', fontsize=28)
plt.xlabel('Northing (m)', fontsize=24)
plt.ylabel('Elevation (m)', fontsize=24)

plt.savefig(f'figures/2dinversion_results_{inversion_title}.png', dpi=300)
