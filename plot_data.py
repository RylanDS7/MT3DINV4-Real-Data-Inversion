# code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import discretize
import matplotlib.pyplot as plt
import copy

with np.load('out/2dinversion_results_P01cf.npz', allow_pickle=True) as data:
    minv_tetm = data['model']
    data_model = data['dpred']
    sigma_est = data['rho_est']
    peris2use = data['peris']
    rx_locs2d = data['rx_locs2d']
    mesh = data['mesh'].item()

mesh = discretize.TensorMesh.deserialize(mesh)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# resistivity
mtrue = np.log10(1/np.exp(sigma_est))
clim = [np.log10(1), np.log10(1e6)]

cmap = copy.copy(plt.get_cmap("Spectral"))
cmap.set_over("white")

dat = mesh.plot_image(
    (mtrue),
    ax=ax,
    # grid=True,
    clim=clim,
    range_x=[rx_locs2d[:,0].min()-500, rx_locs2d[:,0].max()+500],
    range_y=[rx_locs2d[:,1].min()-1500, rx_locs2d[:,1].max()+100],
    pcolor_opts={"cmap": cmap}
)

plt.scatter(rx_locs2d[:,0], rx_locs2d[:,1], color='black', s=50, zorder=5)

ax.set_title('Resistivity')
plt.colorbar(
    dat[0],
    label=r'Resistivity ($\Omega$m)', 
    ticks=[clim[0],clim[1]], 
    format="$10^{%.1f}$", 
    shrink=0.6,
    extend='max'
).ax.tick_params(labelsize=14)

plt.title('2D Inversion Results - P01cf', fontsize=16)
plt.xlabel('Northing (m)', fontsize=14)
plt.ylabel('Elevation (m)', fontsize=14)

plt.savefig('figures/2dinversion_results_P01cf.png', dpi=300)
