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
inversion_title = '1145'

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

rxZ = []
for rx in rxData.values():
    for p in peris:
        freqData = rx.loc[rx['period'] == p]
        Z = np.array([[freqData['z_xx'].values[0], freqData['z_xy'].values[0]], 
                        [freqData['z_yx'].values[0], freqData['z_yy'].values[0]]])
        Z[0, :] = Z[0, :] * _impUnitEDI2SI
        Z[1, :] = Z[0, :] * _impUnitEDI2SI
        rxZ.append(Z)
for Z in rxZ:
    Zdet = np.sqrt(Z[0, 1] * Z[1, 0])
    data_vec.append(Zdet.real)
    data_vec.append(Zdet.imag)

data_vec = np.array(data_vec)

# ==================================================
# Setup survey objects
# ==================================================

rx_loc = np.array([[0.0]])  # 1D surface location

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
thicknesses = np.logspace(-1, 3, n_layers)
mesh = TensorMesh([thicknesses], x0="0")

sigma0 = 1/3000
sigma = np.log(np.ones(n_layers) * sigma0)

mapping = maps.ExpMap(nP=n_layers)

sim = nsem.Simulation1DPrimarySecondary(
    mesh,
    survey=survey,
    sigmaMap=mapping,
    solver=Pardiso,
)

# ==================================================
# Setup the data misfit and regularization
# ==================================================

dmisfit = data_misfit.L2DataMisfit(data=data_obj, simulation=sim)

reg= regularization.WeightedLeastSquares(
    mesh,
    mapping=mapping,
    reference_model=sigma,
)

# set alphas
reg.alpha_s = 1e-4
reg.alpha_x = 1

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

directiveList = [betaest, beta, target]

inv = inversion.BaseInversion(invProb, directiveList=directiveList)
opt.remember('xc')

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
