# code by Rylan Stutters - github.com/RylanDS7

from simpeg import maps, utils, data, optimization, maps, regularization, inverse_problem, directives, inversion, data_misfit
import discretize
import numpy as np
from pymatsolver import Pardiso
from simpeg.electromagnetics import natural_source as nsem
import matplotlib.pyplot as plt
import utm
import mtpy as mt


inversion_title = 'top_line_2d'

# ==================================================
# Load data
# ==================================================

mtc = mt.MTCollection()
mtc.open_collection(inversion_title)
mtd = mtc.to_mt_data()
mtc.close_collection()
mtd.rotate(183) # rotate to strike angle

rxData = {}
for tf in mtd.keys():
    freqDict = {}
    freqs = mtd[tf].Z.frequency
    for i, f in enumerate(freqs):
        freqDict[f] = mtd[tf].Z.z[i]
    rxData[tf] = freqDict

# ==================================================
# Get locations and freqs
# ==================================================

rx_locs = []

for key in mtd.keys():
    rx_locs += [utm.from_latlon(mtd[key].latitude, mtd[key].longitude)[:2] + (mtd[key].elevation,)]
rx_locs = np.array(rx_locs)
rx_locs2d = rx_locs[:, 0::2]

# only use freqs that each reciever has data for
freqs2use = []
for f in mtd.get_periods()**-1:
    freq_count = 0
    for key in mtd.keys():
        if f in mtd[key].Z.frequency:
            freq_count += 1
        if freq_count == mtd.n_stations:
            freqs2use.append(f)


# ==================================================
# Setup mesh
# ==================================================

print("Building mesh")

ncx = 100 # number of core mesh cells
dx = 50 # base cell width
npad_x = 20  # number of padding cells
exp_x = 1.2 # expansion rate of padding cells
ncy = 50    
dy = 40
npad_y = 15
exp_y = 1.2

hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
hy = [(dy, npad_y, -exp_y), (dy, ncy), (dy, npad_y, exp_y)]
hx_cells = discretize.utils.unpack_widths(hx)
hy_cells = discretize.utils.unpack_widths(hy)

x_center = rx_locs[:, 0].mean()
y_surface = rx_locs[:, 2].mean()          

x0 = x_center - hx_cells.sum() / 2
y0 = y_surface - ((dy * ncy) / 3) - (hy_cells.sum() / 2)
mesh = discretize.TensorMesh([hx, hy], origin=[x0, y0])

active_cells = discretize.utils.mesh_utils.active_from_xyz(mesh, rx_locs2d)

print(f"Mesh has {mesh.n_cells} cells")
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
mesh.plot_grid(ax=ax)
ax.scatter(rx_locs[:,0], rx_locs[:, 2], color='orange', s=100, zorder=5)
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

_impUnitEDI2SI = 4 * np.pi * 1e-4

data_vec_te = []
data_vec_tm = []

for freq in freqs2use:
    for rx in rxData.keys(): # real components
        data_vec_tm += [rxData[rx][freq][0, 1].real * _impUnitEDI2SI]
        data_vec_te += [rxData[rx][freq][1, 0].real * _impUnitEDI2SI]
    for rx in rxData.keys(): # imag components
        if freq in rxData[rx].keys():
            data_vec_tm += [rxData[rx][freq][0, 1].imag * _impUnitEDI2SI]
            data_vec_te += [rxData[rx][freq][1, 0].imag * _impUnitEDI2SI]

data_vec_te = np.hstack(data_vec_te)
data_vec_tm = np.hstack(data_vec_tm)

# ==================================================
# Setup the SimPEG survey and simulation
# ==================================================

survey_te = nsem.Survey(src_list_te)
data_obj_te = data.Data(survey_te, data_vec_te)
survey_tm = nsem.Survey(src_list_tm)
data_obj_tm = data.Data(survey_tm, data_vec_tm)

sim_type = "h"
fixed_boundary=True

actmap = maps.InjectActiveCells(

    mesh, indActive=active_cells, valInactive=np.log(1e-8)

)

m0 = (np.ones(mesh.nC) * np.log(1/1e4))[active_cells]

sim_kwargs = {"sigmaMap": maps.ExpMap() * actmap}
test_mod = m0

# create the simulation
sim_tm = nsem.simulation.Simulation2DMagneticField(
    mesh,
    survey=survey_tm,
    **sim_kwargs,
    solver=Pardiso,
)

sim_te = nsem.simulation.Simulation2DElectricField(
    mesh,
    survey=survey_te,
    **sim_kwargs,
    solver=Pardiso,
)

# ==================================================
# Create the data misfits
# ==================================================

std_te = 0.03
std_tm = 0.1

print('[INFO] Getting things started on inversion...')

# TM mode
dmisfit_tm = data_misfit.L2DataMisfit(data=data_obj_tm, simulation=sim_tm)

# TE mode
data_obj_te.standard_deviation = np.abs(data_vec_te) * std_te

dmisfit_te = data_misfit.L2DataMisfit(data=data_obj_te, simulation=sim_te)

# assign the weights
dmisfit_te.W = 1. / (np.abs(data_obj_te.dobs) * std_te + np.percentile(np.abs(data_obj_te.dobs), 5, method='lower'))
dmisfit_tm.W = 1. / (np.abs(data_obj_tm.dobs) * std_tm + np.percentile(np.abs(data_obj_tm.dobs), 10, method='lower'))
dmisfit_combo = dmisfit_tm + dmisfit_te

coolingFactor = 2
coolingRate = 2
beta0_ratio = 1e0

# check for percentile floor

# Map for a regularization
regmap = maps.IdentityMap(nP=int(active_cells.sum()))

reg_tetm = regularization.WeightedLeastSquares(mesh, active_cells=active_cells, mapping=regmap)

# set alpha length scales
reg_tetm.alpha_s = 1e-8
reg_tetm.alpha_x = 1
reg_tetm.alpha_y = 1
reg_tetm.alpha_z = 1

opt_tetm = optimization.ProjectedGNCG(maxIter=20, upper=np.inf, lower=-np.inf)
invProb_tetm = inverse_problem.BaseInvProblem(dmisfit_combo, reg_tetm, opt_tetm)
beta = directives.BetaSchedule(
    coolingFactor=coolingFactor, coolingRate=coolingRate
)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
target = directives.TargetMisfit()

directiveList = [
    beta, 
    betaest, 
    target,
]

inv_tetm = inversion.BaseInversion(
    invProb_tetm, directiveList=directiveList)
# opt.LSshorten = 0.5
opt_tetm.remember('xc')

# ==================================================
# Run the inversion and save the results
# ==================================================

minv_tetm = inv_tetm.run(m0)
data_model = sim_te.dpred(minv_tetm)
rho_est = actmap * minv_tetm

np.save("out/minv_term.npy", minv_tetm)
np.save("out/data_model.npy", data_model)
np.save("out/rho_est.npy", rho_est)