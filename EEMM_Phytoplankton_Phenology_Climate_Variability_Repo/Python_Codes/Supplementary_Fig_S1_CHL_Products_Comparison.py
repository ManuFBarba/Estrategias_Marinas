# -*- coding: utf-8 -*-
"""

######## Supplementary Fig. S1. Intercomparison L3 vs. L4 Chl-a #########

"""

#Loading required libraries
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

# Load datasets
ds_CMEMS = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_SA.nc')
ds_MODIS = xr.open_dataset(r'E:\...\CHL_MODIS_L3_4Km_8day_Data\CLIPPED/CHL_L3_4km_SA.nc')

# Calculate mean chl-a concentration for both datasets
mean_CMEMS = ds_CMEMS.CHL.mean(dim=('lat', 'lon'), skipna=True)
mean_CMEMS = mean_CMEMS.resample(time='1D').mean(skipna=True)
mean_MODIS = ds_MODIS.chlor_a.mean(dim=('lat', 'lon'), skipna=True)

# Find common time points between datasets
common_time = np.intersect1d(mean_CMEMS.time.values, mean_MODIS.time.values)
mean_CMEMS_common = mean_CMEMS.sel(time=common_time)
mean_MODIS_common = mean_MODIS.sel(time=common_time)


# Calculate metrics
def calculate_metrics(predicted, observed):
    # Remove zero values for log scale calculations
    mask = (predicted > 0) & (observed > 0)
    predicted = predicted[mask]
    observed = observed[mask]
    
    # R^2
    mean_predicted = np.mean(predicted)
    ss_total = np.sum((observed - mean_predicted) ** 2)
    ss_residual = np.sum((observed - predicted) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    # Bias
    bias = np.mean(predicted - observed)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(observed, predicted))
    
    return r2, bias, rmse

r2, bias, rmse = calculate_metrics(mean_CMEMS_common, mean_MODIS_common)


mean_CMEMS['time'] = pd.to_datetime(mean_CMEMS.time)
max_time = pd.Timestamp(mean_CMEMS.time.max().item())
years = pd.date_range(start='1998', end=max_time, freq='4YE').year


## Representing plot
fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=1200, gridspec_kw={'width_ratios': [2, 1]})

# Time series
axs[0].set_yscale('log')
axs[0].plot(mean_CMEMS.time, mean_CMEMS, color='k', label='Daily L4 1Km Copernicus-GlobColour', alpha=1)
axs[0].plot(mean_MODIS.time, mean_MODIS, color='green', label='8-day L3m 4Km NASA Aqua-MODIS', alpha=0.8)

axs[0].set_title('South Atlantic (SA)', fontsize=15)
axs[0].set_ylabel(r'[Chl-a] (mg$\cdot$m$^{-3}$)', fontsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=12, length=4, width=1)
axs[0].set_xticks([pd.Timestamp(str(year)) for year in years])  
axs[0].set_xticklabels([str(year) for year in years], fontsize=12)  
axs[0].set_yticks([0.1, 0.25, 0.5, 1, 2.5, 5])
axs[0].yaxis.set_major_formatter(plt.ScalarFormatter())
axs[0].set_ylim(0.15, 5)
# axs[0].legend(frameon=False, loc='upper left', fontsize=14)

# Validation Scatter plot
sc = axs[1].scatter(mean_CMEMS_common, mean_MODIS_common, c=mean_CMEMS_common.time.dt.year, cmap='Spectral_r', alpha=0.7)
axs[1].set_xscale('log')
axs[1].set_yscale('log')

# Plot 1:1 line
min_val = min(np.min(mean_CMEMS_common), np.min(mean_MODIS_common))
max_val = max(np.max(mean_CMEMS_common), np.max(mean_MODIS_common))
axs[1].plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.8)

# Add metrics to the scatter plot
metrics_text = (f'R²: {r2:.2f}\nδ: {bias:.2f}\nRMSE: {rmse:.2f}')
axs[1].text(0.65, 0.25, metrics_text, transform=axs[1].transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1, edgecolor='none', facecolor='white'))

axs[1].set_title('South Atlantic (SA)', fontsize=15)
axs[1].set_xlabel(r'C-GC [Chl-a] (mg$\cdot$m$^{-3}$)', fontsize=14)
axs[1].set_ylabel(r'MODIS [Chl-a] (mg$\cdot$m$^{-3}$)', fontsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=12, length=4, width=1)
axs[1].set_xticks([0.1, 0.25, 0.5, 1, 2.5, 5])
axs[1].xaxis.set_major_formatter(plt.ScalarFormatter())
axs[1].set_yticks([0.1, 0.25, 0.5, 1, 2.5, 5])
axs[1].yaxis.set_major_formatter(plt.ScalarFormatter())
axs[1].set_xlim(0.15, 5)
axs[1].set_ylim(0.15, 5)

# cbar = plt.colorbar(sc, ticks=np.arange(2003, 2025, 2), label='')
# cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.show()

outfile = r'E:\...\Figures\Fig_S1\Fig_S1gh.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')

