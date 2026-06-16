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

data_dir = 'profileData/'
inversion_title = 'P01cf'

# ==================================================
# Load data
# ==================================================
directory_path = Path("./data_corrected")
stations2invert = np.arange(1145, 1151, 1)

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
rx_locs2d = rx_locs[:, [1,2]]

# only use freqs that each reciever has data for
peris2use = []
for p in mtd.get_periods():
    peri_count = 0
    for rx in rxData.values():
        if p in rx['period'].values:
            peri_count += 1
        if peri_count == mtd.n_stations and p**-1 > 5 and p**-1 < 1000:
            peris2use.append(p)


# ==================================================
# Setup mesh
# ==================================================

print("Building mesh")

x_center = rx_locs2d[:, 0].mean()
y_surface = rx_locs2d[:, 1].mean()  

x_width = rx_locs2d[:, 0].max() - rx_locs2d[:, 0].min()

dx = 40 # base cell width
ncx = int((x_width + 1000) / dx) # number of core mesh cells
npad_x = 20  # number of padding cells
exp_x = 1.5 # expansion rate of padding cells

dy = 40 # base cell width
ncy = 150 # number of core mesh cells
npad_y = 20 # number of padding cells
exp_y = 1.5 # expansion rate of padding cells

hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
hy = [(dy, npad_y, -exp_y), (dy, ncy)]
hx_cells = discretize.utils.unpack_widths(hx)
hy_cells = discretize.utils.unpack_widths(hy)        

x0 = x_center - hx_cells.sum() / 2
y0 = y_surface - hy_cells.sum() + 200
mesh = discretize.TensorMesh([hx, hy], origin=[x0, y0])

active_cells = discretize.utils.mesh_utils.active_from_xyz(mesh, rx_locs2d)

# drape rxs to mesh surface
rx_locs2d = utils.shift_to_discrete_topography(mesh, rx_locs2d, active_cells)

print(f"Mesh has {mesh.n_cells} cells")
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# ax.set_xlim(rx_locs2d[:, 0].min() - 500, rx_locs2d[:, 0].max() + 500)
# ax.set_ylim(y_surface - 1000, y_surface + 300)
# mesh.plot_grid(ax=ax)
# ax.scatter(rx_locs2d[:,0], rx_locs2d[:, 1], color='orange', s=100, zorder=5)
# plt.show()

# ==================================================
# Setup SimPEG sources and rxs
# ==================================================

src_list_te = []
src_list_tm = []

for p in peris2use:
    rx_list_te = [
        nsem.receivers.Impedance(rx_locs2d, orientation="xy", component="real"),
        nsem.receivers.Impedance(rx_locs2d, orientation="xy", component="imag"),
    ]

    rx_list_tm = [
        nsem.receivers.Impedance(rx_locs2d, orientation="yx", component="real"),
        nsem.receivers.Impedance(rx_locs2d, orientation="yx", component="imag"),
    ]

    src_list_te.append(nsem.sources.Planewave(rx_list_te, frequency=p**-1))
    src_list_tm.append(nsem.sources.Planewave(rx_list_tm, frequency=p**-1))

# ==================================================
# Setup data objects
# ==================================================

_impUnitEDI2SI = 4 * np.pi * 1e-4

data_vec_te = []
data_vec_tm = []

for p in peris2use:
    rxZ = []
    for rx in rxData.values():
        freqData = rx.loc[rx['period'] == p]
        Z = np.array([[freqData['z_xx'].values[0], freqData['z_xy'].values[0]], 
                      [freqData['z_yx'].values[0], freqData['z_yy'].values[0]]]) * _impUnitEDI2SI
        rxZ.append(Z)
    for Z in rxZ:
        data_vec_tm.append(Z[0, 1].real)
        data_vec_te.append(Z[1, 0].real) 
    for Z in rxZ:
        data_vec_tm.append(Z[0, 1].imag)
        data_vec_te.append(Z[1, 0].imag)

data_vec_te = np.array(data_vec_te)
data_vec_tm = np.array(data_vec_tm)

# ==================================================
# Setup the SimPEG survey and simulation
# ==================================================

survey_te = nsem.Survey(src_list_te)
data_obj_te = data.Data(survey_te, data_vec_te)
survey_tm = nsem.Survey(src_list_tm)
data_obj_tm = data.Data(survey_tm, data_vec_tm)

actmap = maps.InjectActiveCells(
    mesh, active_cells=active_cells, value_inactive=np.log(1e-8)
)
expmap = maps.ExpMap()

m0 = (np.ones(mesh.nC) * np.log(1/680))[active_cells]
mapping = expmap * actmap

model = np.ones(mesh.nC) * 1e-8
model[active_cells] = 1/680

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
mesh_plot = mesh.plot_image(model, ax=ax, grid=True, cmap='viridis', 
                            range_x=(rx_locs2d[:, 0].min() - 500, rx_locs2d[:, 0].max() + 500),
                            range_y=(y_surface - 1000, y_surface + 300))
cb = plt.colorbar(mesh_plot[0], ax=ax, orientation='vertical')
cb.set_label('Conductivity (S/m)')
ax.scatter(rx_locs2d[:,0], rx_locs2d[:, 1], color='orange', s=100, zorder=5)
plt.show()

# create the simulation
sim_tm = nsem.simulation.Simulation2DMagneticField(
    mesh,
    survey=survey_tm,
    sigmaMap=mapping,
    solver=Pardiso,
)

sim_te = nsem.simulation.Simulation2DElectricField(
    mesh,
    survey=survey_te,
    sigmaMap=mapping,
    solver=Pardiso,
)

# ==================================================
# Create the data misfits
# ==================================================

print('Getting things started on inversion...')

floor = 0.03
percent = 0.05

data_obj_te.standard_deviation = np.abs(data_vec_te) * percent + floor
data_obj_tm.standard_deviation = np.abs(data_vec_tm) * percent + floor

dmisfit_tm = data_misfit.L2DataMisfit(data=data_obj_tm, simulation=sim_tm)
dmisfit_te = data_misfit.L2DataMisfit(data=data_obj_te, simulation=sim_te)

dmisfit_combo = dmisfit_tm + dmisfit_te

# Map for a regularization
regmap = maps.IdentityMap(nP=int(active_cells.sum()))

reg_tetm = regularization.WeightedLeastSquares(
    mesh,
    active_cells=active_cells,
    mapping=regmap,
    reference_model=m0,
)

# set alpha length scales
reg_tetm.alpha_s = 1
reg_tetm.alpha_x = 1
reg_tetm.alpha_y = 1

opt_tetm = optimization.ProjectedGNCG(maxIter=20, upper=np.inf, lower=-np.inf)
invProb_tetm = inverse_problem.BaseInvProblem(dmisfit_combo, reg_tetm, opt_tetm)

coolingFactor = 2
coolingRate = 2
beta0_ratio = 1e0

beta = directives.BetaSchedule(
    coolingFactor=coolingFactor, coolingRate=coolingRate
)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
target = directives.TargetMisfit()
save_model = directives.SaveOutputDictEveryIteration(on_disk=True, directory=f".\out\{inversion_title}_models")

directiveList = [betaest, beta, target, save_model]

inv_tetm = inversion.BaseInversion(invProb_tetm, directiveList=directiveList)
opt_tetm.remember('xc')


# Check forward simulation of halfspace model
dpred_te = sim_te.dpred(m0)
dpred_tm = sim_tm.dpred(m0)

np.save("out/data_te.npy", data_vec_te)
np.save("out/data_tm.npy", data_vec_tm)

r_te = (data_vec_te - dpred_te) / data_obj_te.standard_deviation
r_tm = (data_vec_tm - dpred_tm) / data_obj_tm.standard_deviation

ind = np.arange(len(r_te))
step = mtd.n_stations * 2
for i in np.arange(step):
    plt.figure()
    plt.plot(ind[i::step], dpred_te[i::step], 'x-', label='TE Predicted')
    plt.plot(ind[i::step], dpred_tm[i::step], 'x-', label='TM Predicted')
    plt.plot(ind[i::step], data_vec_te[i::step], 's-', label='TE Observed')
    plt.plot(ind[i::step], data_vec_tm[i::step], 's-', label='TM Observed')
    if step - i > mtd.n_stations:
        plt.title(f"Real Impedance, Station {i}")
    else:
        plt.title(f"Imag Impedance, Station {i - mtd.n_stations}")     
    plt.legend()
    plt.show()

# ==================================================
# Run the inversion and save the results
# ==================================================

minv_tetm = inv_tetm.run(m0)
data_model = sim_te.dpred(minv_tetm)
rho_est = actmap * minv_tetm

save_mesh = mesh.serialize()

np.savez(
    f"out/2dinversion_results_{inversion_title}.npz",
    model=minv_tetm,
    dpred=data_model,
    rho_est=rho_est,
    peris=peris2use,
    rx_locs2d=rx_locs2d,
    mesh=save_mesh,
)