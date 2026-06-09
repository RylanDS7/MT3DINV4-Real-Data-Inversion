# code by Rylan Stutters - github.com/RylanDS7

import mtpy as mt
import numpy as np
from pathlib import Path

directory_path = Path("./data_corrected")

inversion_title = 'profileData/P01cf'
stations2invert = np.append(np.arange(1130, 1140, 1), np.array([1338]))

print(f"Stations to invert: {stations2invert}")

mtc = mt.MTCollection()
mtc.open_collection(inversion_title)

for file_path in directory_path.iterdir():
    station_num = int(file_path.stem[2:6])
    if station_num in stations2invert:
        mt_object = mt.MT()
        mt_object.read(file_path)
        mt_object.survey_metadata.id = 'survey'
        mt_object.station = f'{station_num}'
        mt_object.station_metadata.id = f'{station_num}'
        mt_object.tf_id = f'{station_num}'
        mtc.add_tf(mt_object)

mtc.close_collection()