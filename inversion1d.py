"""
Created on Wed Jun 17 15:54:23 2026

@author: Rylan Stuttters - github.com/RylanDS7
"""

from simpeg import maps, data, optimization, regularization, inverse_problem, directives, inversion, data_misfit, utils
from discretize import TensorMesh
import numpy as np
from pymatsolver import Pardiso
from simpeg.electromagnetics import natural_source as nsem
import matplotlib.pyplot as plt
from pathlib import Path
import utm
from mtpy import MTData
from mtpy.core.mt import MT

data_dir = 'profileData/'
inversion_title = '1140'

# ==================================================
# Load data
# ==================================================
directory_path = Path("./data_corrected")
stations2invert = [1145]

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
# Get locations
# ==================================================

rx_locs = []
elevations = []

for rx in rxData.values():
    elevations.append(rx['elevation'].iloc[0])

for rx in rxData.values():
    east, north = utm.from_latlon(rx['latitude'].iloc[0], rx['longitude'].iloc[0])[:2]
    rx_locs.append([east, north, np.mean(elevations)])


# ==================================================
# Setup data objects
# ==================================================

_impUnitEDI2SI = 4 * np.pi * 1e-4

data_vec = []
peris = rx['period'].tolist()
peris = np.array(peris)
peris = peris[(peris >= 0.001) & (peris <= 1)]

rxZ = []
for rx in rxData.values():
    for p in peris:
        freqData = rx.loc[rx['period'] == p]
        Z = np.array([[freqData['z_xx'].values[0], freqData['z_xy'].values[0]], 
                        [freqData['z_yx'].values[0], freqData['z_yy'].values[0]]])
        Z = Z * _impUnitEDI2SI
        rxZ.append(Z)
for Z in rxZ:
    data_vec.append(Z[1,0].real)
    data_vec.append(Z[1,0].imag)

data_vec = np.array(data_vec)

rxZ = np.array(rxZ)
fig, axes = plt.subplots(2,2)
axes = axes.flatten()

axes[0].plot(rxZ[:, 0, 0].real, label="Zxx Real")
axes[0].plot(rxZ[:, 0, 0].imag, label="Zxx Imag")
axes[0].legend()
axes[1].plot(rxZ[:, 0, 1].real, label="Zxy Real")
axes[1].plot(rxZ[:, 0, 1].imag, label="Zxy Imag")
axes[1].legend()
axes[2].plot(rxZ[:, 1, 0].real, label="Zyx Real")
axes[2].plot(rxZ[:, 1, 0].imag, label="Zyx Imag")
axes[2].legend()
axes[3].plot(rxZ[:, 1, 1].real, label="Zyy Real")
axes[3].plot(rxZ[:, 1, 1].imag, label="Zyy Imag")
axes[3].legend()

plt.show()

# ==================================================
# Setup survey objects
# ==================================================

rx_loc = np.array([[-0.1]])  # 1D surface location

src_list = []

for p in peris:

    rx_list = [
        nsem.receivers.Impedance(
            locations_e=rx_loc,
            orientation="xy",
            component="real",
        ),
        nsem.receivers.Impedance(
            locations_e=rx_loc,
            orientation="xy",
            component="imag",
        ),
    ]

    src_list.append(
        nsem.sources.Planewave(
            receiver_list=rx_list,
            frequency=p**-1,
        )
    )

survey = nsem.Survey(src_list)
data_obj = data.Data(survey, data_vec)

floor = 0.03
percent = 0.05
data_obj.standard_deviation = np.abs(data_vec) * percent + floor

# ==================================================
# Setup simulation
# ==================================================

n_layers = 1000
thicknesses = np.logspace(1, -1, n_layers)
mesh = TensorMesh([thicknesses], origin='N')

print("Mesh z extent:", mesh.nodes_x.min(), "to", mesh.nodes_x.max())

sigma0 = 1/3000
sigma = np.log(np.ones(mesh.nC) * sigma0)

mapping = maps.ExpMap()

sim = nsem.Simulation1DElectricField(
    mesh,
    survey=survey,
    sigmaMap=mapping,
    solver=Pardiso,
)

fig, ax = plt.subplots(figsize=(8, 2))
mesh.plot_grid(ax=ax, nodes=True, centers=True, colors="k")

ax.set_title("1D Discretized Mesh")
ax.set_xlabel("X coordinate")
plt.show()

# ==================================================
# Setup the data misfit and regularization
# ==================================================

dmisfit = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)

reg_map = maps.IdentityMap(nP=mesh.nC)
reg= regularization.WeightedLeastSquares(
    mesh,
    mapping=reg_map,
    reference_model=sigma,
)

# set alphas
reg.alpha_s = 1e-4
reg.alpha_x = 1

opt = optimization.ProjectedGNCG(maxIter=20, maxIterLS=20, maxIterCG=30, tolCG=1e-3)
invProb = inverse_problem.BaseInvProblem(dmisfit, reg, opt)

coolingFactor = 2
coolingRate = 2
beta0_ratio = 1e0

beta = directives.BetaSchedule(
    coolingFactor=coolingFactor, coolingRate=coolingRate
)
betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio)
target = directives.TargetMisfit()

directiveList = [betaest, beta, target]

inv = inversion.BaseInversion(invProb, directiveList=directiveList)
opt.remember('xc')

dpred = sim.dpred(sigma)

ind = np.arange(len(dpred))
step = 2
for i in np.arange(step):
    plt.figure()
    plt.plot(ind[i::step], dpred[i::step], 'x-', label='Predicted')
    plt.plot(ind[i::step], data_vec[i::step], 's-', label='Observed')
    if step - i > mtd.n_stations:
        plt.title(f"Real Impedance, Station {i}")
    else:
        plt.title(f"Imag Impedance, Station {i - mtd.n_stations}")     
    plt.legend()
    plt.show()

# ==================================================
# Run the inversion and save the results
# ==================================================

minv = inv.run(sigma)
data_model = sim.dpred(minv)

save_mesh = mesh.serialize()

np.savez(
    f"out/1dinversion_results_{inversion_title}.npz",
    model=minv,
    dpred=data_model,
    peris=peris,
    mesh=save_mesh,
)
