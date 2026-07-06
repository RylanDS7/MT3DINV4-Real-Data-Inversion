# code by Rylan Stutters - github.com/RylanDS7

from simpeg import maps, data, optimization, regularization, inverse_problem, directives, inversion, data_misfit, utils
import discretize
import numpy as np
from pymatsolver import Pardiso
from simpeg.electromagnetics import natural_source as nsem
import matplotlib.pyplot as plt
from pathlib import Path
import utm
from mtpy import MTData
from mtpy.core.mt import MT

stations2invert = np.arange(1130, 1184, 1)

# ==================================================
# Load data
# ==================================================

data_dir = 'profileData/'
directory_path = Path("./data_corrected")

print(f"Stations to invert: {stations2invert}")

# Build list of MT objects for selected stations
mt_objects = []
for file_path in directory_path.iterdir():
    if file_path.suffix.lower() != '.edi':
        continue
    station_num = int(file_path.stem[2:6])
    if station_num in stations2invert:
        mt_obj = MT()
        mt_obj.read(file_path)
        mt_obj.survey_metadata.id = 'survey'
        mt_obj.station = f'{station_num}'
        mt_obj.station_metadata.id = f'{station_num}'
        mt_obj.tf_id = f'{station_num}'
        mt_objects.append(mt_obj)

# Load into MTData
mtd = MTData()
mtd.add_station(mt_objects)
print(f"Number of stations loaded: {mtd.n_stations}")

mdf = mtd.to_dataframe()

rxData = {}
for rx in stations2invert:
    sdf = mdf.loc[mdf['station'] == str(rx)]
    rxData[rx] = sdf


# ==================================================
# Get locations and freqs
# ==================================================

rx_locs = []
elevations = []

for rx in rxData.values():
    elevations.append(rx['elevation'].iloc[0])

for rx in rxData.values():
    east, north = utm.from_latlon(rx['latitude'].iloc[0], rx['longitude'].iloc[0])[:2]
    rx_locs.append([east, north, np.mean(elevations)])

rx_locs = np.array(rx_locs)

# freqs to use for 3d inversion
freqs2use = [8.0566, 23.439, 52.748, 99.634, 234.4, 433.64, 984.4099999999999]


# ==================================================
# Setup mesh
# ==================================================

print("Building mesh")

x_center = rx_locs[:, 0].mean()
y_center = rx_locs[:, 1].mean()  

x_width = rx_locs[:, 0].max() - rx_locs[:, 0].min()
y_width = rx_locs[:, 1].max() - rx_locs[:, 1].min()

dx = 50 # base cell width
ncx = int((x_width + 1000) / dx) # number of core mesh cells
npad_x = 10  # number of padding cells
exp_x = 1.5 # expansion rate of padding cells

dy = 50 # base cell width
ncy = int((y_width + 1000) / dx) # number of core mesh cells
npad_y = 10 # number of padding cells
exp_y = 1.5 # expansion rate of padding cells

dz = 25 # base cell width
ncz = 40 # number of core mesh cells
npad_z = 10 # number of padding cells
exp_z = 1.5 # expansion rate of padding cells

hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
hy = [(dy, npad_y, -exp_y), (dy, ncy), (dx, npad_x, exp_x)]
hz = [(dz, npad_z, -exp_z), (dz, ncz)]
hx_cells = discretize.utils.unpack_widths(hx)
hy_cells = discretize.utils.unpack_widths(hy)     
hz_cells = discretize.utils.unpack_widths(hz)    

x0 = x_center - hx_cells.sum() / 2
y0 = y_center - hy_cells.sum() / 2
z0 = np.mean(elevations) - hz_cells.sum() + 100

mesh = discretize.TensorMesh([hx, hy, hz], origin=[x0, y0, z0])

active_cells = discretize.utils.mesh_utils.active_from_xyz(mesh, rx_locs)

# drape rxs to mesh surface
rx_locs = utils.shift_to_discrete_topography(mesh, rx_locs, active_cells)

print(f"Mesh has {mesh.n_cells} cells")

# Core region extents (ignoring padding cells), with a little buffer
core_xmin = x_center - x_width/2 - 100
core_xmax = x_center + x_width/2 + 100
core_ymin = y_center - y_width/2 - 100
core_ymax = y_center + y_width/2 + 100

z_target = np.mean(rx_locs[:, 2])
core_zmin = z_target - 300
core_zmax = z_target + 100

# Slice indices closest to receiver/core center
x_ind = np.argmin(np.abs(mesh.cell_centers_x - x_center))
y_ind = np.argmin(np.abs(mesh.cell_centers_y - y_center))
z_ind = np.argmin(np.abs(mesh.cell_centers_z - z_target))

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# --- X-slice (Y-Z plane) ---
mesh.plot_slice(active_cells, normal="X", ind=x_ind, ax=ax[0], grid=True,
                 pcolor_opts={"cmap": "Greys", "alpha": 0.3})
ax[0].scatter(rx_locs[:, 1], rx_locs[:, 2], c="red", s=15, zorder=5, label="Rx")
ax[0].set_xlim(core_ymin, core_ymax)
ax[0].set_ylim(core_zmin, core_zmax)
ax[0].set_title("X-slice (Y-Z, core zoom)")
ax[0].legend()

# --- Y-slice (X-Z plane) ---
mesh.plot_slice(active_cells, normal="Y", ind=y_ind, ax=ax[1], grid=True,
                 pcolor_opts={"cmap": "Greys", "alpha": 0.3})
ax[1].scatter(rx_locs[:, 0], rx_locs[:, 2], c="red", s=15, zorder=5, label="Rx")
ax[1].set_xlim(core_xmin, core_xmax)
ax[1].set_ylim(core_zmin, core_zmax)
ax[1].set_title("Y-slice (X-Z, core zoom)")
ax[1].legend()

# --- Z-slice (X-Y plane) ---
mesh.plot_slice(active_cells, normal="Z", ind=z_ind, ax=ax[2], grid=True,
                 pcolor_opts={"cmap": "Greys", "alpha": 0.3})
ax[2].scatter(rx_locs[:, 0], rx_locs[:, 1], c="red", s=15, zorder=5, label="Rx")
ax[2].set_xlim(core_xmin, core_xmax)
ax[2].set_ylim(core_ymin, core_ymax)
ax[2].set_title("Z-slice (X-Y, core zoom)")
ax[2].set_aspect("equal")
ax[2].legend()

plt.tight_layout()
plt.show()


# ==================================================
# Setup SimPEG sources and rxs
# ==================================================

src_list = []

for f in freqs2use:
    rx_list = [
        nsem.receivers.Impedance(rx_locs, orientation="xy", component="real"),
        nsem.receivers.Impedance(rx_locs, orientation="xy", component="imag"),
        nsem.receivers.Impedance(rx_locs, orientation="yx", component="real"),
        nsem.receivers.Impedance(rx_locs, orientation="yx", component="imag"),
    ]

    src_list.append(nsem.sources.PlanewaveXYPrimary(rx_list, frequency=f))


# ==================================================
# Setup data objects
# ==================================================

_impUnitEDI2SI = 4 * np.pi * 1e-4

data_vec = []

for f in freqs2use:
    xy_real, xy_imag, yx_real, yx_imag = [], [], [], []
    for rx in rxData.values():
        freqData = rx.loc[np.isclose(rx['period'], f**-1, rtol=0.01)]
        Z = np.array([[freqData['z_xx'].values[0], freqData['z_xy'].values[0]],
                      [freqData['z_yx'].values[0], freqData['z_yy'].values[0]]]) * _impUnitEDI2SI
        xy_real.append(Z[0, 1].real)
        xy_imag.append(Z[0, 1].imag)
        yx_real.append(Z[1, 0].real)
        yx_imag.append(Z[1, 0].imag)
    data_vec.extend(yx_real)
    data_vec.extend(yx_imag)
    data_vec.extend(xy_real)
    data_vec.extend(xy_imag)

data_vec = np.array(data_vec)


# ==================================================
# Setup the SimPEG survey and simulation
# ==================================================

survey = nsem.Survey(src_list)
data_obj = data.Data(survey, data_vec)

actmap = maps.InjectActiveCells(
    mesh, active_cells=active_cells, value_inactive=np.log(1e-8)
)
expmap = maps.ExpMap()

sigma_background = 1/500

m0 = (np.ones(mesh.nC) * np.log(sigma_background))[active_cells]
mapping = expmap * actmap

model = np.ones(mesh.nC) * 1e-8
model[active_cells] = sigma_background

# create the simulation
sim = nsem.simulation.Simulation3DPrimarySecondary(
    mesh,
    survey=survey,
    sigmaMap=mapping,
    sigmaPrimary=model,
    solver=Pardiso,
)


# ==================================================
# Create the data misfits
# ==================================================

floor = 0.03
percent = 0.05

data_obj.standard_deviation = np.abs(data_vec) * percent + floor

dmisfit = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)

# Map for a regularization
regmap = maps.IdentityMap(nP=int(active_cells.sum()))

reg = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    mapping=regmap,
    reference_model=m0,
)

# set alpha length scales
reg.alpha_s = 1e-5
reg.alpha_x = 1
reg.alpha_y = 1
reg.alpha_z = 1

opt = optimization.ProjectedGNCG(maxIter=20, upper=np.inf, lower=-np.inf)
invProb = inverse_problem.BaseInvProblem(dmisfit, reg, opt)

coolingFactor = 2
coolingRate = 2
beta0_ratio = 1e0

beta = directives.BetaSchedule(
    coolingFactor=coolingFactor, coolingRate=coolingRate
)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
target = directives.TargetMisfit()
# save_model = directives.SaveOutputDictEveryIteration(on_disk=True, directory=f".\out\{inversion_title}_models")

directiveList = [betaest, beta, target]

inv = inversion.BaseInversion(invProb, directiveList=directiveList)
opt.remember('xc')

