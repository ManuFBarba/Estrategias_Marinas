# -*- coding: utf-8 -*-
"""

############### Fig. 2. Methods of Phenology Metrics ###################

"""

#Loading required Python modules
import numpy as np
import xarray as xr 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


## Loading lat and lon arrays from 'CHL_Phenology_metrics.py'

## Loading previously-processed CHL dataset
ds_CHL_SA = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_SA.nc')


## Fig. 2. Methodology of Phenology Indices
ds_year = ds_CHL_SA.sel(time=slice("2012-01-01", "2012-12-31"))
ds_year_smoothed = ds_year.rolling(time=24, center=True, min_periods=1).mean()
pixel_series = ds_year_smoothed.sel(lat=36.5, lon=-6.5, method='nearest').CHL.values.squeeze()
threshold = np.nanmedian(pixel_series) * 1.05

#Bloom ini
intersection_ini_idx = np.where(pixel_series >= threshold)[0][0]
intersection_ini_time = ds_year.time[intersection_ini_idx].values
intersection_ini_value = pixel_series[intersection_ini_idx]
#Bloom fin
after_start_idx = (intersection_ini_idx + np.where(pixel_series[intersection_ini_idx:] < threshold)[0][0]) - 1
intersection_fin_time = ds_year.time[after_start_idx].values
intersection_fin_value = pixel_series[after_start_idx]
# Bloom maximum
max_value_idx = np.nanargmax(pixel_series)
max_value_time = ds_year.time[max_value_idx].values
max_value = pixel_series[max_value_idx]

time_range = ds_year.time.values

fig = plt.figure(figsize=(14, 6))
plt.plot(ds_year.time, pixel_series, color='darkgreen', label= 'Copernicus-GlobColour Chl-a')
plt.plot([time_range[0], time_range[-1]], [threshold, threshold], color='#B92049', linestyle='--', linewidth=2, label='Threshold [5% median]')
plt.fill_between(ds_year.time, threshold, pixel_series, where=(pixel_series > threshold), color='green', alpha=0.5, label='Phytoplankton Bloom')
plt.scatter(max_value_time, max_value, color='#B92049', s=175, marker='*', zorder=5, label='Bloom Maximum Chl-a')
plt.scatter(intersection_ini_time, threshold, color='#FED380', s=100, marker='s', zorder=5, label='Bloom Initiation')
plt.scatter(intersection_fin_time, threshold, color='#545CA8', s=100, marker='s', zorder=5, label='Bloom Termination')
plt.vlines(max_value_time, ymin=0.3, ymax=max_value, color='purple', linestyle=':', linewidth=1.5)
plt.scatter(max_value_time, 0.3, color='purple', s=125, marker='^', zorder=5, clip_on=False, label='Bloom Peak')

# Bloom duration
arrow_y_position = threshold - 0.05
plt.annotate(
    '', xy=(intersection_ini_time, arrow_y_position), xytext=(intersection_fin_time, arrow_y_position),
    arrowprops=dict(arrowstyle='<->', color='black', lw=2)
)

ini_num = mdates.date2num(intersection_ini_time)
fin_num = mdates.date2num(intersection_fin_time)
mid_point = (ini_num + fin_num) / 2

plt.text(mid_point, arrow_y_position - 0.1, 'Bloom Duration', 
         fontsize=15, color='black', ha='center')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.gca().set_xticklabels([
    r'$\mathrm{Jan}_{y}$', r'$\mathrm{Mar}_{y}$', r'$\mathrm{May}_{y}$', r'$\mathrm{Jul}_{y}$', 
    r'$\mathrm{Sep}_{y}$', r'$\mathrm{Nov}_{y}$', r'$\mathrm{Jan}_{y+1}$'
], fontsize=15)

plt.ylim(0.3, 2)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel('[Chl-a] [mg·m$^{-3}$]', fontsize=15)
plt.legend(fontsize=15, loc='upper left', frameon=False, bbox_to_anchor=(-0.01, 1.02))
plt.show()

outfile = r'E:\...\Figures\Fig_2\Fig_2.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')

