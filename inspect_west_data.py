import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

freqs2use = [8.0566, 23.439, 52.748, 99.634, 234.4, 433.64, 984.4099999999999]

for freq in freqs2use:
    ax.axvline(x=freq, color='b', linestyle='--', linewidth=1)

ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Station ID')
ax.set_yticks(stations2invert)
ax.set_title('Station Available Frequencies')
plt.savefig(f'{fig_dir}freq_plot.png')


# ==================================================
# Adjust Freqencies to align
# ==================================================

fig, ax = plt.subplots(figsize=(18, 14))

inversion_freq = set()

for key in mtd.station_paths:
    station = mtd.get_station(key)
    freq = 1 / station.period.to_numpy()

    plot_freq = []
    for f in freq:
        if any(abs(freqs2use - f) < 0.01 * f):
            plot_freq.append(f)
            inversion_freq.add(f)

    try:
        assert(len(plot_freq) == len(freqs2use))
    except:
        print(f"Station {station} missing {len(freqs2use) - len(plot_freq)} frequencies")

    id = np.ones(len(plot_freq)) * int(station.station)

    ax.scatter(plot_freq, id, c='blue', s=1.5)

for freq in freqs2use:
    ax.axvline(x=freq, color='b', linestyle='--', linewidth=1)

ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Station ID')
ax.set_yticks(stations2invert)
ax.set_title('Station Frequencies to Invert')
plt.savefig(f'{fig_dir}freq2use_plot.png')


# ==================================================
# Plot Impedance Components
# ==================================================

_impUnitEDI2SI = 4 * np.pi * 1e-4

mdf = mtd.to_dataframe()

rxData = {}
for rx in stations2invert:
    sdf = mdf.loc[mdf['station'] == str(rx)]
    rxData[rx] = sdf

with PdfPages(f"{fig_dir}impedances2invert.pdf") as pdf:
    for key in rxData.keys():
        rx = rxData[key]
        periods = rx['period'].unique()
        Z_arr = []
        for p in periods:
            freqData = rx.loc[rx['period'] == p]
            Z = np.array([[freqData['z_xx'].values[0], freqData['z_xy'].values[0]], 
                        [freqData['z_yx'].values[0], freqData['z_yy'].values[0]]]) * _impUnitEDI2SI
            Z_arr.append(Z)

        Z_arr = np.array(Z_arr)

        fig, axes = plt.subplots(2, 2, figsize=(18,14))
        axes = axes.flatten()

        axes[0].scatter(periods**-1, Z_arr[:, 0, 0], s=3)
        axes[0].set_xscale('log')
        axes[0].set_xlabel("Frequency")
        axes[0].set_title("Z_xx")

        axes[1].scatter(periods**-1, Z_arr[:, 0, 1], s=3)
        axes[1].set_xscale('log')
        axes[1].set_xlabel("Frequency")
        axes[1].set_title("Z_xy")

        axes[2].scatter(periods**-1, Z_arr[:, 1, 0], s=3)
        axes[2].set_xscale('log')
        axes[2].set_xlabel("Frequency")
        axes[2].set_title("Z_yx")

        axes[3].scatter(periods**-1, Z_arr[:, 1, 1], s=3)
        axes[3].set_xscale('log')
        axes[3].set_xlabel("Frequency")
        axes[3].set_title("Z_yy")

        for ax in axes:
            for freq in freqs2use:
                ax.axvline(x=freq, color='b', linestyle='--', linewidth=1)

        fig.suptitle(f"MT Response Rx {key}")

        pdf.savefig(fig)
        plt.close()



# ==================================================
# Plot Phase Tensor Maps for inversion freqs
# ==================================================

with PdfPages(f"{fig_dir}phase_tensor_maps.pdf") as pdf:
    for f in freqs2use:
        try:
            ptm = mtd.plot_phase_tensor_map(plot_period=1/f, ellipse_size=0.001, arrow_size=0.001, fig_size=(18,14))
            pdf.savefig(ptm.fig)
            plt.close(ptm.fig)
        except Exception as e:
            print(f"Skipping freq {f}: {e}")



# ==================================================
# Plot Resistivity and Phase Maps for inversion freqs
# ==================================================

with PdfPages(f"{fig_dir}phase_resistivity_maps.pdf") as pdf:
    for f in freqs2use:
        try:
            prm = mtd.plot_resistivity_phase_maps(plot_period=1/f, plot_xx=True, plot_yy=True, plot_det=False, fig_size=(18,14))
            pdf.savefig(prm.fig)
            plt.close(prm.fig)
        except Exception as e:
            print(f"Skipping freq {f}: {e}")



        

        
