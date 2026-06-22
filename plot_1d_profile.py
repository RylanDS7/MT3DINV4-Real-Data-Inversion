# code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import discretize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from simpeg import utils

inversion_titles = [1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150]


fig = plt.figure(figsize=(8, 9))
ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])

resistivity = []

for inversion in inversion_titles:
    with np.load(f'out/1dinversion_results_{inversion}.npz', allow_pickle=True) as data:
        minv = data['model']
        data_model = data['dpred']
        peris2use = data['peris']
        mesh = data['mesh'].item()

    mesh = discretize.TensorMesh.deserialize(mesh)

    mtrue = 1/np.exp(minv)

    resistivity.append(mtrue)

    cell_lengths = np.flip(mesh.edge_x_lengths)
    mtrue = np.flip(mtrue)


    utils.plot_1d_layer_model(cell_lengths, mtrue, ax=ax1, label=f'{inversion}')

ax1.set_xlim([0.1, 10000])
ax1.set_ylim([0, 1000])
ax1.set_title("Inverted Model")
ax1.yaxis.set_inverted(True)

plt.legend()
plt.xlabel('Resistivity ($\Omega$m)')

plt.show()

resistivity = np.array(resistivity)
resistivity = resistivity.T

n_depth, n_stations = resistivity.shape

z_edges = mesh.nodes  # depth edges (n_depth + 1)

cmap = cm.turbo
norm = LogNorm(vmin=np.nanmin(resistivity), vmax=np.nanmax(resistivity))

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.8  # controls spacing between stations

for i in range(n_stations):
    for j in range(n_depth):
        color = cmap(norm(resistivity[j, i]))

        rect = Rectangle(
            (i - bar_width/2, z_edges[j]),     # x, y bottom-left
            bar_width,                         # width
            z_edges[j+1] - z_edges[j],         # height
            facecolor=color,
            edgecolor='none'
        )
        ax.add_patch(rect)

ax.set_xlim(-1, n_stations)
ax.set_ylim(z_edges[0], z_edges[-1])
ax.invert_yaxis()

ax.set_xticks(range(n_stations))
ax.set_xticklabels([f"S{i+1}" for i in range(n_stations)])

ax.set_xlabel("Station")
ax.set_ylabel("Depth (m)")
ax.set_title("1D Inversions as Discrete Station Bars")

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
plt.colorbar(sm, ax=ax, label="Resistivity ($\Omega$m)")

plt.savefig("figures/1dinversion_results_P01cf.png")
plt.show()