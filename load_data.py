"""
Created on Fri Jun 12 11:34:23 2026

@author: Rylan Stuttters - github.com/RylanDS7
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from mtpy import MTData
from mtpy.core.mt import MT

data_dir = 'profileData/'
inversion_title = 'P01cf'

directory_path = Path("./data_corrected")
stations2invert = np.arange(1140, 1151, 1)

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

with PdfPages(f'out/{inversion_title}_mt_response.pdf') as pdf:
    for key in mtd.station_paths:
        plot_obj = mtd.plot_mt_response(key)
        fig = plot_obj.fig
        pdf.savefig(fig)
        plt.clf()
        plt.close(fig)
        
with PdfPages(f'out/{inversion_title}_phase_tensor.pdf') as pdf:
    for key in mtd.station_paths:
        plot_obj = mtd.plot_phase_tensor(key)
        fig = plot_obj.fig
        pdf.savefig(fig)
        plt.clf()
        plt.close(fig)
    

