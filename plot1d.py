# code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import discretize
import matplotlib.pyplot as plt
from simpeg import utils

with np.load('out/1dinversion_results_1140.npz', allow_pickle=True) as data:
    minv = data['model']
    data_model = data['dpred']
    peris2use = data['peris']
    mesh = data['mesh'].item()

mesh = discretize.TensorMesh.deserialize(mesh)

mtrue = 1/np.exp(minv)

cell_lengths = np.flip(mesh.edge_x_lengths)
mtrue = np.flip(mtrue)

fig = plt.figure(figsize=(8, 9))
x_min = np.min(minv)
x_max = np.max(minv)

ax1 = fig.add_axes([0.2, 0.15, 0.7, 0.7])
utils.plot_1d_layer_model(cell_lengths, mtrue, ax=ax1)

ax1.set_title("Inverted Model")
plt.xlabel('Resistivity ($\Omega$m)')

plt.show()