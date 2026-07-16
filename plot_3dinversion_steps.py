# code by Rylan Stutters - github.com/RylanDS7
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from simpeg import maps
import discretize
from pathlib import Path
import numpy as np
from mtpy import MTData
from mtpy.core.mt import MT

inversion = '3dinversion_results_02'
inversion_dir = Path(f'./out/{inversion}')
pdf_filename = f"figures/{inversion}/misfits"

# ==================================================
# Load mtpy data
# ==================================================

stations2invert = np.arange(1130, 1184, 1)

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

mdf = mtd.to_dataframe()

rxData = {}
for rx in stations2invert:
    sdf = mdf.loc[mdf['station'] == str(rx)]
    rxData[rx] = sdf


_impUnitEDI2SI = 4 * np.pi * 1e-4

data_vec = []

# freqs to use for 3d inversion
freqs2use = [8.0566, 23.439, 52.748, 99.634, 234.4, 433.64, 984.4099999999999]

for f in freqs2use:
    xy_real, xy_imag, yx_real, yx_imag = [], [], [], []
    for rx in rxData.values():
        freqData = rx.loc[np.isclose(rx['period'], f**-1, rtol=0.01)]
        Z = np.array([[freqData['z_xx'].values[0], freqData['z_xy'].values[0]],
                      [freqData['z_yx'].values[0], freqData['z_yy'].values[0]]]) * _impUnitEDI2SI
        xy_real.append(Z[0, 1].real)
        xy_imag.append(Z[0, 1].imag)
        yx_real.append(Z[1, 0].real)
        yx_imag.append(Z[1, 0].imag)
    data_vec.extend(yx_real)
    data_vec.extend(yx_imag)
    data_vec.extend(xy_real)
    data_vec.extend(xy_imag)

data_vec = np.array(data_vec)


# ==================================================
# Compare misfits
# ==================================================

def unpack_dpred(dpred):
    data = dpred.reshape(len(freqs2use), 4, len(rxData))
    return data.transpose(2, 1, 0)

raw_data = unpack_dpred(data_vec)

for file in inversion_dir.iterdir():
    if not file.stem.startswith("InversionModel"):
        continue
    output = np.load(file, allow_pickle=True)["arr_0"].item()

    iter = output['iter']
    dpred = output['dpred']

    plot_data = unpack_dpred(dpred)

    # Data type labels mapping to your loop order
    type_labels = ["xy_real", "xy_imag", "yx_real", "yx_imag"]
    rx_names = list(rxData.keys())  # To label each subplot by its rx name


    with PdfPages(pdf_filename+str(iter)+".pdf") as pdf:
    # Loop over each data type (Each loop iteration = A new PDF page)
        for type_idx in range(4):
            ncols = 8
            nrows = int(np.ceil(len(rxData) / ncols))
            
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8), sharex=True)
            fig.suptitle(f"Data Type: {type_labels[type_idx]}", fontsize=16, fontweight='bold')
            
            # Flatten axes array for easy 1D looping, in case it's a 2D matrix
            axes_flat = axes.flatten() if len(rxData) > 1 else [axes]
            
            # Plot data for each receiver on this page
            for rx_idx in range(len(rxData)):
                ax = axes_flat[rx_idx]
                
                # Extract y-values for this specific rx and data type across all freqs
                y_values = plot_data[rx_idx, type_idx, :]
                y_values_raw = raw_data[rx_idx, type_idx, :]

                
                # Plot against frequencies (X-axis)
                line_sim, = ax.plot(freqs2use, y_values, marker='o', linestyle='-', label='Simulated')
                line_data, = ax.plot(freqs2use, y_values_raw, marker='o', linestyle='-', label='Data')
                if rx_idx == 0:
                    line_handles = [line_sim, line_data]

                # Format individual subplots
                ax.set_title(f"Receiver: {rx_names[rx_idx]}", fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # Log scale is common for frequency data; remove if you prefer linear
                ax.set_xscale('log') 
                
                # Add labels only to outer edge plots to reduce clutter
                if rx_idx >= (nrows - 1) * ncols:
                    ax.set_xlabel("Frequency (Hz)")
            
            total_slots = nrows * ncols
            if total_slots > len(rxData):
                # 1. Use the very first blank axis slot for the legend
                legend_ax = axes_flat[len(rxData)]
                legend_ax.axis('off')  # Hide the box outlines, ticks, and labels
                
                # 2. Draw the legend centrally inside this blank boundary box
                legend_ax.legend(handles=line_handles, loc='center', fontsize=12, frameon=True, shadow=True)
                
                # 3. Hide any remaining empty subplots if there are more than one
                for blank_ax in axes_flat[len(rxData) + 1:]:
                    blank_ax.set_visible(False)
                
            plt.tight_layout()
            
            # Save current figure as a new page in the PDF, then close it to free memory
            pdf.savefig(fig)
            plt.close(fig)

