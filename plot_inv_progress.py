"""
Created on Mon Jun 15 11:19:21 2026

@author: Rylan Stuttters - github.com/RylanDS7
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from simpeg import maps
import discretize
from pathlib import Path
import numpy as np
import copy

data_path = Path("./out/P01cf_models")

with np.load('out/2dinversion_results_P01cf.npz', allow_pickle=True) as data:
    minv_tetm = data['model']
    data_model = data['dpred']
    sigma_est = data['sigma_est']
    peris2use = data['peris']
    rx_locs2d = data['rx_locs2d']
    mesh = data['mesh'].item()

mesh = discretize.TensorMesh.deserialize(mesh)

active_cells = discretize.utils.mesh_utils.active_from_xyz(mesh, rx_locs2d)
actmap = maps.InjectActiveCells(
    mesh, active_cells=active_cells, value_inactive=np.log(1e-8)
)

data_vec_te = np.load("out/data_te.npy")
data_vec_tm = np.load("out/data_tm.npy")

model_pdf = PdfPages("out/model_plots.pdf")
misfit_pdf = PdfPages("out/misfit_plots.pdf")

for file in data_path.iterdir():
    output = np.load(file, allow_pickle=True)["arr_0"].item()

    sigma_est = actmap * output['m']

    # ---- model plot ----
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

    ax.set_title('2D Inversion Results - P01cf', fontsize=16)
    ax.set_xlabel('Northing (m)', fontsize=14)
    ax.set_ylabel('Elevation (m)', fontsize=14)

    model_pdf.savefig(fig)
    plt.close(fig)

    # ---- misfit plots: 12 subplots per page (one page per iteration) ----
    dpred = output['dpred']
    l = len(dpred) // 2
    dpred_te = dpred[:l]
    dpred_tm = dpred[l:]
    ind = np.arange(l)
    step = 6 * 2

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i in range(step):
        ax = axes[i]
        ax.plot(ind[i::step], dpred_te[i::step], 'x-', label='TE Predicted')
        ax.plot(ind[i::step], dpred_tm[i::step], 'x-', label='TM Predicted')
        ax.plot(ind[i::step], data_vec_te[i::step], 'o-', label='TE Observed')
        ax.plot(ind[i::step], data_vec_tm[i::step], 'o-', label='TM Observed')

        if step - i > 6:
            ax.set_title(f"Real Impedance, Station {i}")
        else:
            ax.set_title(f"Imag Impedance, Station {i - 6}")

        ax.legend(fontsize=7)

    fig.suptitle(f"Misfit Plots - {file.name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    misfit_pdf.savefig(fig)
    plt.close(fig)

model_pdf.close()
misfit_pdf.close()
