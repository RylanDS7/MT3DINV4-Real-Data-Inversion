# code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import discretize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

inversion_dir = '3dinversion_results_02/'

with np.load(f'out/{inversion_dir}final_model.npz', allow_pickle=True) as data:
    model = data['model']
    dpred = data['dpred']
    freqs = data['freqs']
    rx_locs = data['rx_locs']
    mesh = data['mesh'].item()
    active_cells = data['active_cells']

mesh = discretize.TensorMesh.deserialize(mesh)

# Expand model to full mesh
full_model = np.full(mesh.n_cells, np.nan)
full_model[active_cells] = np.log10(1/np.exp(model))

# rx_locs is expected to be an (n_rx, 3) array: columns x, y, z
y_min, y_max = rx_locs[:, 1].min(), rx_locs[:, 1].max()
z_min, z_max = rx_locs[:, 2].min() - 1000, rx_locs[:, 2].max()

# Add padding so the zoom isn't cut exactly at the receiver extents
# (e.g. 20% of the range, with a fallback for near-zero ranges)
y_range = y_max - y_min
z_range = z_max - z_min
y_pad = 0.2 * y_range if y_range > 0 else 50
z_pad = 0.2 * z_range if z_range > 0 else 50

x_centers = mesh.cell_centers_x
x_inds = np.arange(len(x_centers))

pdf_path = f'figures/{inversion_dir}model_slices.pdf'
os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

with PdfPages(pdf_path) as pdf:
    for x_ind in x_inds:
        fig, ax = plt.subplots(figsize=(9, 7))

        out = mesh.plot_slice(
            full_model,
            normal='X',
            ind=x_ind,
            ax=ax,
            grid=True,
            pcolor_opts={'cmap': 'viridis'},
        )
        plt.colorbar(out[0], ax=ax, label='Log10(Resistivity)')

        ax.set_xlim(y_min - y_pad, y_max + y_pad)
        ax.set_ylim(z_min - z_pad, z_max + z_pad)

        ax.scatter(rx_locs[:, 1], rx_locs[:, 2], c='red', s=15, marker='v', label='Receivers')
        ax.legend(loc='upper right')

        ax.set_title(f'Model Slice — X = {x_centers[x_ind]:.1f} m')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        plt.tight_layout()

        pdf.savefig(fig)   # append this figure as a new page
        plt.close(fig)     # free memory before the next iteration