import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mtpy import MTData
from mtpy.core.mt import MT

fig_dir = "./figures/west_data/"
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


# ==================================================
# Plot Stations
# ==================================================

station_plot = mtd.plot_stations()

ax = station_plot.ax
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
xpad = (xmax - xmin) * 0.1
ypad = (ymax - ymin) * 0.05
ax.set_xlim(xmin - xpad, xmax + xpad)
ax.set_ylim(ymin - ypad, ymax + ypad)

station_plot.save_plot(f'{fig_dir}station_plot.png')


# ==================================================
# Plot Freqencies
# ==================================================

fig, ax = plt.subplots(figsize=(18, 14))

unique_freqs = set()

for key in mtd.station_paths:
    station = mtd.get_station(key)
    freq = 1 / station.period.to_numpy()
    unique_freqs.update(freq)

    id = np.ones(len(freq)) * int(station.station)

    ax.scatter(freq, id, c='blue', s=1.5)

freqs2use = [8.0566, 12.452, 23.439, 52.748, 99.634, 433.64, 984.4099999999999]

for freq in freqs2use:
    ax.axvline(x=freq, color='b', linestyle='--', linewidth=1)

ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Station ID')
ax.set_yticks(stations2invert)
ax.set_title('Station Available Frequencies')
plt.savefig(f'{fig_dir}freq_plot.png')
plt.show()
