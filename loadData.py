# code by Rylan Stutters - github.com/RylanDS7

import mtpy as mt
from pathlib import Path

directory_path = Path("./data_corrected")

inversion_title = 'P10cf'
# stations2invert = [1216, 1227, 1238, 1249, 1260, 1271, 1282, 1293, 1301, 1312]
stations2invert = [1232, 1233, 1234, 1235, 1236, 1237, 1238]

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