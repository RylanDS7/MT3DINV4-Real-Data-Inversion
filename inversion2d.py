# code by Rylan Stutters - github.com/RylanDS7

from simpeg import maps, utils, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
import numpy as np
from pymatsolver import Pardiso
from simpeg.electromagnetics import natural_source as nsem
import matplotlib.pyplot as plt
import utm
import mtpy as mt

inversion_title = 'profileData/P01cf'

# ==================================================
# Load data
# ==================================================

mtc = mt.MTCollection()
mtc.open_collection(inversion_title)
mtd = mtc.to_mt_data()
mtc.close_collection()
mtd.rotate(90)

_impUnitEDI2SI = 4 * np.pi * 1e-4

rxData = {}
for i, key in enumerate(mtd.keys()):
    rx = mtd[key]
    freqs = rx.Z.frequency

    freqData = {}
    for ii, f in enumerate(freqs):
        freqData[f] = mtd[key].impedance[ii].values * _impUnitEDI2SI

    rxData[key] = freqData

# ==================================================
# Get locations and freqs
# ==================================================

rx_locs = []

for key in mtd.keys():
    rx_locs += [utm.from_latlon(mtd[key].latitude, mtd[key].longitude)[:2] + (mtd[key].elevation,)]

rx_locs = np.array(rx_locs)
rx_locs2d = rx_locs[:, [1,2]]

# only use freqs that each reciever has data for
freqs2use = []
for f in mtd.get_periods()**-1:
    freq_count = 0
    for key in mtd.keys():
        if f in mtd[key].Z.frequency:
            freq_count += 1
        if freq_count == mtd.n_stations and f < 1000:
            freqs2use.append(f)


# ==================================================
# Setup mesh
# ==================================================

print("Building mesh")

x_center = rx_locs2d[:, 0].mean()
y_surface = rx_locs2d[:, 1].mean()  

x_width = rx_locs2d[:, 0].max() - rx_locs2d[:, 0].min()

dx = 40 # base cell width
ncx = int((x_width + 500) / dx) # number of core mesh cells
npad_x = 20  # number of padding cells
exp_x = 1.5 # expansion rate of padding cells

dy = 40 # base cell width
ncy = 150 # number of core mesh cells
npad_y = 20 # number of padding cells
exp_y = 1.5 # expansion rate of padding cells

hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
hy = [(dy, npad_y, -exp_y), (dy, ncy), (dy, npad_y, exp_y)]
hx_cells = discretize.utils.unpack_widths(hx)
hy_cells = discretize.utils.unpack_widths(hy)        

x0 = x_center - hx_cells.sum() / 2
y0 = y_surface - (hy_cells.sum() / 2) - (dy * ncy / 3)
mesh = discretize.TensorMesh([hx, hy], origin=[x0, y0])

active_cells = discretize.utils.mesh_utils.active_from_xyz(mesh, rx_locs2d)

print(f"Mesh has {mesh.n_cells} cells")
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
mesh.plot_grid(ax=ax)
ax.scatter(rx_locs2d[:,0], rx_locs2d[:, 1], color='orange', s=100, zorder=5)
plt.show()

# ==================================================
# Setup SimPEG sources and rxs
# ==================================================

src_list_te = []
src_list_tm = []

for f in freqs2use:
    rx_list_te = [
        nsem.receivers.Impedance(rx_locs2d, orientation="xy", component="real"),
        nsem.receivers.Impedance(rx_locs2d, orientation="xy", component="imag"),
    ]

    rx_list_tm = [
        nsem.receivers.Impedance(rx_locs2d, orientation="yx", component="real"),
        nsem.receivers.Impedance(rx_locs2d, orientation="yx", component="imag"),
    ]

    src_list_te.append(nsem.sources.Planewave(rx_list_te, frequency=f))
    src_list_tm.append(nsem.sources.Planewave(rx_list_tm, frequency=f))

# ==================================================
# Setup data objects
# ==================================================

data_vec_te = []
data_vec_tm = []

for src in src_list_te:
    for tf in mtd.keys():
        data_vec_te.append(rxData[tf][src.frequency][1, 0].real)
    for tf in mtd.keys():
        data_vec_te.append(rxData[tf][src.frequency][1, 0].imag)

for src in src_list_tm:
    for tf in mtd.keys():
        data_vec_tm.append(rxData[tf][src.frequency][0, 1].real)
    for tf in mtd.keys():
        data_vec_tm.append(rxData[tf][src.frequency][0, 1].imag)

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
    mesh, indActive=active_cells, valInactive=np.log(1e-8)
)
expmap = maps.ExpMap()



m0 = (np.ones(mesh.nC) * np.log(1/1e3))[active_cells]
mapping = expmap * actmap

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

print('[INFO] Getting things started on inversion...')

floor = 0.05  # prevents over-weighting small values

data_obj_te.standard_deviation = np.abs(data_vec_te) * 0.05 + floor
data_obj_tm.standard_deviation = np.abs(data_vec_tm) * 0.1 + floor

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
reg_tetm.alpha_s = 1e-8
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

directiveList = [betaest, beta, target]

inv_tetm = inversion.BaseInversion(invProb_tetm, directiveList=directiveList)
opt_tetm.remember('xc')


# Check forward simulation of halfspace model
dpred_te = sim_te.dpred(m0)
dpred_tm = sim_tm.dpred(m0)

r_te = (data_vec_te - dpred_te) / data_obj_te.standard_deviation
r_tm = (data_vec_tm - dpred_tm) / data_obj_tm.standard_deviation

print(np.max(np.abs(r_te)))
print(np.max(np.abs(r_tm)))
print(np.median(np.abs(r_te)))
print(np.median(np.abs(r_tm)))

ind = np.arange(len(r_te))
plt.figure()
plt.plot(ind, dpred_te, 'x-', label='TE Predicted')
plt.plot(ind, dpred_tm, 'x-', label='TM Predicted')
plt.plot(ind, data_vec_te, 's-', label='TE Observed')
plt.plot(ind, data_vec_tm, 's-', label='TM Observed')
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
    freqs=freqs2use,
    rx_locs2d=rx_locs2d,
    mesh=save_mesh,
)