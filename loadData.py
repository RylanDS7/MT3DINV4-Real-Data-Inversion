# code by Rylan Stutters - github.com/RylanDS7

import numpy as np
from pathlib import Path

# fix ssl error on windows for mtpy
import ssl
_original = ssl.SSLContext.load_default_certs
def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _original(self, purpose)
    except ssl.SSLError:
        pass
ssl.SSLContext.load_default_certs = _safe_load_default_certs

from mtpy import MTData
from mtpy.core.mt import MT

directory_path = Path("./data_corrected")
stations2invert = np.arange(1140, 1151, 1)

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
print(f"Station keys: {list(mtd.station_paths)}")

# Check first station
first_key = list(mtd.keys())[0]
station = mtd.get_station(first_key)
breakpoint()
print(f"\nFirst station: {station.station}")
print(f"Frequencies: {station.Z.frequency}")
print(f"Z shape: {station.Z.z.shape}")