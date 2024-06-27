# -*- coding: utf-8 -*-
"""

#########################      Figure 7 in 
Fernández-Barba, M., Huertas, I. E., & Navarro, G. (2024). 
Assessment of surface and bottom marine heatwaves along the Spanish coast. 
Ocean Modelling, 190, 102399.                          ########################

"""

#Loading required libraries
import numpy as np
import xarray as xr 

import matplotlib.pyplot as plt

import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FormatStrFormatter

from scipy.stats import spearmanr


## Load MHWs_from_MATLAB.py
## Load bathymetry data from Datos_ESMARES.py


## Mask out where elevation > 2500 m
elevation_NA_masked = xr.where(elevation_NA > 2500, np.NaN, elevation_NA)
elevation_AL_masked = xr.where(elevation_AL > 2500, np.NaN, elevation_AL)
elevation_CAN_masked = xr.where(elevation_CAN > 2500, np.NaN, elevation_CAN)
elevation_SA_masked = xr.where(elevation_SA > 2500, np.NaN, elevation_SA)
elevation_BAL_masked = xr.where(elevation_BAL > 2500, np.NaN, elevation_BAL)


## Maximum MLT
# Max_MLT_SA = MLT_SA.max(dim='time', skipna=True)
# Max_MLT_AL = MLT_AL.max(dim='time', skipna=True)
# Max_MLT_BAL = MLT_BAL.max(dim='time', skipna=True)
# Max_MLT_NA = MLT_NA.max(dim='time', skipna=True)
# Max_MLT_CAN = MLT_CAN.max(dim='time', skipna=True)

# #Save data#
# Max_MLT_SA = Max_MLT_SA.values
# np.save(r'...\Max_MLT_SA.npy', Max_MLT_SA)
# Max_MLT_AL = Max_MLT_AL.values
# np.save(r'...\Max_MLT_AL.npy', Max_MLT_AL)
# Max_MLT_BAL = Max_MLT_BAL.values
# np.save(r'...\Max_MLT_BAL.npy', Max_MLT_BAL)
# Max_MLT_NA = Max_MLT_NA.values
# np.save(r'...\Max_MLT_NA.npy', Max_MLT_NA)
# Max_MLT_CAN = Max_MLT_CAN.values
# np.save(r'...\Max_MLT_CAN.npy', Max_MLT_CAN)

#Load Max MLT data#
Max_MLT_SA = np.load(r'...\Max_MLT_SA.npy')
Max_MLT_AL = np.load(r'...\Max_MLT_AL.npy')
Max_MLT_BAL = np.load(r'...\Max_MLT_BAL.npy')
Max_MLT_NA = np.load(r'...\Max_MLT_NA.npy')
Max_MLT_CAN = np.load(r'...\Max_MLT_CAN.npy')



## MLT/Bottom Depth

#Interpolate Bathymetry grid to the GLORYS dataset grid
#Elevation_Interp
new_lat_SA = np.asarray(ds_Model_SA.latitude)
new_lon_SA = np.asarray(ds_Model_SA.longitude)
elevation_SA_interp = elevation_SA_masked.interp(lat=new_lat_SA, lon=new_lon_SA)
elevation_SA_interp = elevation_SA_interp.rename({'lat': 'latitude'})
elevation_SA_interp = elevation_SA_interp.rename({'lon': 'longitude'})

new_lat_AL = np.asarray(ds_Model_AL.latitude)
new_lon_AL = np.asarray(ds_Model_AL.longitude)
elevation_AL_interp = elevation_AL_masked.interp(lat=new_lat_AL, lon=new_lon_AL)
elevation_AL_interp = elevation_AL_interp.rename({'lat': 'latitude'})
elevation_AL_interp = elevation_AL_interp.rename({'lon': 'longitude'})

new_lat_BAL = np.asarray(ds_Model_BAL.latitude)
new_lon_BAL = np.asarray(ds_Model_BAL.longitude)
elevation_BAL_interp = elevation_BAL_masked.interp(lat=new_lat_BAL, lon=new_lon_BAL)
elevation_BAL_interp = elevation_BAL_interp.rename({'lat': 'latitude'})
elevation_BAL_interp = elevation_BAL_interp.rename({'lon': 'longitude'})

new_lat_NA = np.asarray(ds_Model_NA.latitude)
new_lon_NA = np.asarray(ds_Model_NA.longitude)
elevation_NA_interp = elevation_NA_masked.interp(lat=new_lat_NA, lon=new_lon_NA)
elevation_NA_interp = elevation_NA_interp.rename({'lat': 'latitude'})
elevation_NA_interp = elevation_NA_interp.rename({'lon': 'longitude'})

new_lat_CAN = np.asarray(ds_Model_CAN.latitude)
new_lon_CAN = np.asarray(ds_Model_CAN.longitude)
elevation_CAN_interp = elevation_CAN_masked.interp(lat=new_lat_CAN, lon=new_lon_CAN)
elevation_CAN_interp = elevation_CAN_interp.rename({'lat': 'latitude'})
elevation_CAN_interp = elevation_CAN_interp.rename({'lon': 'longitude'})

#MLT / Bathymetry
MLT_Bottom_SA = (Max_MLT_SA/elevation_SA_interp).T
MLT_Bottom_AL = (Max_MLT_AL/elevation_AL_interp).T
MLT_Bottom_BAL = (Max_MLT_BAL/elevation_BAL_interp).T
MLT_Bottom_NA = (Max_MLT_NA/elevation_NA_interp).T
MLT_Bottom_CAN = (Max_MLT_CAN/elevation_CAN_interp).T

# Convert DataArray to NumPy Array
MLT_Bottom_NA = MLT_Bottom_NA.values
MLT_Bottom_AL = MLT_Bottom_AL.values
MLT_Bottom_CAN = MLT_Bottom_CAN.values
MLT_Bottom_SA = MLT_Bottom_SA.values
MLT_Bottom_BAL = MLT_Bottom_BAL.values


MLT_Bottom_NA = np.where(MLT_Bottom_NA >= 1, 1, MLT_Bottom_NA)
MLT_Bottom_AL = np.where(MLT_Bottom_AL >= 1, 1, MLT_Bottom_AL)
MLT_Bottom_CAN = np.where(MLT_Bottom_CAN >= 1, 1, MLT_Bottom_CAN)
MLT_Bottom_SA = np.where(MLT_Bottom_SA >= 1, 1, MLT_Bottom_SA)
MLT_Bottom_BAL = np.where(MLT_Bottom_BAL >= 1, 1, MLT_Bottom_BAL)



###############################
## 2D Probability Histograms ##
###############################

##North Atlantic
fig, (axs1, axs2, axs3) = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

## Surface / Bottom Total Annual MHW Days ##
Td_NA = MHW_td_NA_MODEL/BMHW_td_NA_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_NA) & ~np.isnan(Td_NA)
MLT_Bottom_NA_clean = MLT_Bottom_NA[valid_indices]
Td_NA_clean = Td_NA[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_NA_clean,
    Td_NA_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

cmap=plt.cm.hot_r
vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
cs1 = axs1.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs1.set_xlabel(r'MLT / Bathy', fontsize=14)
axs1.set_ylabel(r'Total Annual SMHW / BMHW Days', fontsize=14)
axs1.set_title(r'North Atlantic (NA)', fontsize=14)
axs1.xaxis.set_minor_locator(AutoMinorLocator())
axs1.yaxis.set_minor_locator(AutoMinorLocator())
axs1.minorticks_on()
axs1.grid(which='both', linestyle='-', linewidth=0.5)



## BMHW/SMHW Max Intensity ##
MaxInt_NA = BMHW_max_NA_MODEL/MHW_max_NA_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_NA) & ~np.isnan(MaxInt_NA)
MLT_Bottom_NA_clean = MLT_Bottom_NA[valid_indices]
MaxInt_NA_clean = MaxInt_NA[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_NA_clean,
    MaxInt_NA_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im2 = axs2.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs2.set_xlabel(r'MLT / Bathy', fontsize=14)
axs2.set_ylabel(r'BMHW / SMHW Max Intensity', fontsize=14)
axs2.set_title('North Atlantic (NA)', fontsize=14)
axs2.xaxis.set_minor_locator(AutoMinorLocator())
axs2.yaxis.set_minor_locator(AutoMinorLocator())
axs2.minorticks_on()
axs2.grid(which='both', linestyle='-', linewidth=0.5)


## BMHW/SMHW Cum Intensity ##
CumInt_NA = BMHW_cum_NA_MODEL/MHW_cum_NA_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_NA) & ~np.isnan(CumInt_NA)
MLT_Bottom_NA_clean = MLT_Bottom_NA[valid_indices]
CumInt_NA_clean = CumInt_NA[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_NA_clean,
    CumInt_NA_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im3 = axs3.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs3.set_xlabel(r'MLT / Bathy', fontsize=14)
axs3.set_ylabel(r'BMHW / SMHW Cum Intensity', fontsize=14)
axs3.set_title('North Atlantic (NA)', fontsize=14)
axs3.xaxis.set_minor_locator(AutoMinorLocator())
axs3.yaxis.set_minor_locator(AutoMinorLocator())
axs3.minorticks_on()
axs3.grid(which='both', linestyle='-', linewidth=0.5)

# Create a colorbar with 10 intervals
# divider = make_axes_locatable(axs3)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(cs1, cax=cax, ticks=np.linspace(vmin, vmax, num=10))
# cbar.set_label(r'Probability density', fontsize=14)
# cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer tick labels
# cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format tick labels as float with 2 decimal places
# cbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])


# Calculate Spearman correlation for each subplot
corr1, _ = spearmanr(MLT_Bottom_NA_clean, Td_NA_clean)
corr2, _ = spearmanr(MLT_Bottom_NA_clean, MaxInt_NA_clean)
corr3, _ = spearmanr(MLT_Bottom_NA_clean, CumInt_NA_clean)

# Annotate each subplot with the Spearman correlation value
axs1.annotate(f'{corr1:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs2.annotate(f'{corr2:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs3.annotate(f'{corr3:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')

plt.tight_layout()


outfile = r'...\Fig_7\NA.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




##SoG and Alboran Sea
fig, (axs1, axs2, axs3) = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

## Surface / Bottom Total Annual MHW Days ##
Td_AL = MHW_td_AL_MODEL/BMHW_td_AL_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_AL) & ~np.isnan(Td_AL)
MLT_Bottom_AL_clean = MLT_Bottom_AL[valid_indices]
Td_AL_clean = Td_AL[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_AL_clean,
    Td_AL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

cmap=plt.cm.hot_r
vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
cs1 = axs1.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs1.set_xlabel(r'MLT / Bathy', fontsize=14)
axs1.set_ylabel(r'Total Annual SMHW / BMHW Days', fontsize=14)
axs1.set_title(r'SoG and Alboran Sea (AL)', fontsize=14)
axs1.xaxis.set_minor_locator(AutoMinorLocator())
axs1.yaxis.set_minor_locator(AutoMinorLocator())
axs1.minorticks_on()
axs1.grid(which='both', linestyle='-', linewidth=0.5)



## BMHW/SMHW Max Intensity ##
MaxInt_AL = BMHW_max_AL_MODEL/MHW_max_AL_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_AL) & ~np.isnan(MaxInt_AL)
MLT_Bottom_AL_clean = MLT_Bottom_AL[valid_indices]
MaxInt_AL_clean = MaxInt_AL[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_AL_clean,
    MaxInt_AL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im2 = axs2.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs2.set_xlabel(r'MLT / Bathy', fontsize=14)
axs2.set_ylabel(r'BMHW / SMHW Max Intensity', fontsize=14)
axs2.set_title('SoG and Alboran Sea (AL)', fontsize=14)
axs2.xaxis.set_minor_locator(AutoMinorLocator())
axs2.yaxis.set_minor_locator(AutoMinorLocator())
axs2.minorticks_on()
axs2.grid(which='both', linestyle='-', linewidth=0.5)


## BMHW/SMHW Cum Intensity ##
CumInt_AL = BMHW_cum_AL_MODEL/MHW_cum_AL_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_AL) & ~np.isnan(CumInt_AL)
MLT_Bottom_AL_clean = MLT_Bottom_AL[valid_indices]
CumInt_AL_clean = CumInt_AL[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_AL_clean,
    CumInt_AL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im3 = axs3.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs3.set_xlabel(r'MLT / Bathy', fontsize=14)
axs3.set_ylabel(r'BMHW / SMHW Cum Intensity', fontsize=14)
axs3.set_title('SoG and Alboran Sea (AL)', fontsize=14)
axs3.xaxis.set_minor_locator(AutoMinorLocator())
axs3.yaxis.set_minor_locator(AutoMinorLocator())
axs3.minorticks_on()
axs3.grid(which='both', linestyle='-', linewidth=0.5)

# # Create a colorbar with 10 intervals
# divider = make_axes_locatable(axs3)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = plt.colorbar(cs1, cax=cax, ticks=np.linspace(vmin, vmax, num=10))
# cbar.set_label(r'Probability density', fontsize=14)
# cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer tick labels
# cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format tick labels as float with 2 decimal places
# cbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])


# Calculate Spearman correlation for each subplot
corr1, _ = spearmanr(MLT_Bottom_AL_clean, Td_AL_clean)
corr2, _ = spearmanr(MLT_Bottom_AL_clean, MaxInt_AL_clean)
corr3, _ = spearmanr(MLT_Bottom_AL_clean, CumInt_AL_clean)

# Annotate each subplot with the Spearman correlation value
axs1.annotate(f'{corr1:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs2.annotate(f'{corr2:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs3.annotate(f'{corr3:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')

plt.tight_layout()


outfile = r'...\Fig_7\AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




##Canary 
fig, (axs1, axs2, axs3) = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

## Surface / Bottom Total Annual MHW Days ##
Td_CAN = MHW_td_CAN_MODEL/BMHW_td_CAN_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_CAN) & ~np.isnan(Td_CAN)
MLT_Bottom_CAN_clean = MLT_Bottom_CAN[valid_indices]
Td_CAN_clean = Td_CAN[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_CAN_clean,
    Td_CAN_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

cmap=plt.cm.hot_r
vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
cs1 = axs1.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs1.set_xlabel(r'MLT / Bathy', fontsize=14)
axs1.set_ylabel(r'Total Annual SMHW / BMHW Days', fontsize=14)
axs1.set_title(r'Canary (CAN)', fontsize=14)
axs1.xaxis.set_minor_locator(AutoMinorLocator())
axs1.yaxis.set_minor_locator(AutoMinorLocator())
axs1.minorticks_on()
axs1.grid(which='both', linestyle='-', linewidth=0.5)



## BMHW/SMHW Max Intensity ##
MaxInt_CAN = BMHW_max_CAN_MODEL/MHW_max_CAN_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_CAN) & ~np.isnan(MaxInt_CAN)
MLT_Bottom_CAN_clean = MLT_Bottom_CAN[valid_indices]
MaxInt_CAN_clean = MaxInt_CAN[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_CAN_clean,
    MaxInt_CAN_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im2 = axs2.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs2.set_xlabel(r'MLT / Bathy', fontsize=14)
axs2.set_ylabel(r'BMHW / SMHW Max Intensity', fontsize=14)
axs2.set_title('Canary (CAN)', fontsize=14)
axs2.xaxis.set_minor_locator(AutoMinorLocator())
axs2.yaxis.set_minor_locator(AutoMinorLocator())
axs2.minorticks_on()
axs2.grid(which='both', linestyle='-', linewidth=0.5)


## BMHW/SMHW Cum Intensity ##
CumInt_CAN = BMHW_cum_CAN_MODEL/MHW_cum_CAN_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_CAN) & ~np.isnan(CumInt_CAN)
MLT_Bottom_CAN_clean = MLT_Bottom_CAN[valid_indices]
CumInt_CAN_clean = CumInt_CAN[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_CAN_clean,
    CumInt_CAN_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im3 = axs3.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs3.set_xlabel(r'MLT / Bathy', fontsize=14)
axs3.set_ylabel(r'BMHW / SMHW Cum Intensity', fontsize=14)
axs3.set_title('Canary (CAN)', fontsize=14)
axs3.xaxis.set_minor_locator(AutoMinorLocator())
axs3.yaxis.set_minor_locator(AutoMinorLocator())
axs3.minorticks_on()
axs3.grid(which='both', linestyle='-', linewidth=0.5)
# Create a colorbar with 10 intervals
divider = make_axes_locatable(axs3)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(cs1, cax=cax, ticks=np.linspace(vmin, vmax, num=10))
cbar.set_label(r'Probability density', fontsize=14)
cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer tick labels
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format tick labels as float with 2 decimal places
cbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])


# Calculate Spearman correlation for each subplot
corr1, _ = spearmanr(MLT_Bottom_CAN_clean, Td_CAN_clean)
corr2, _ = spearmanr(MLT_Bottom_CAN_clean, MaxInt_CAN_clean)
corr3, _ = spearmanr(MLT_Bottom_CAN_clean, CumInt_CAN_clean)

# Annotate each subplot with the Spearman correlation value
axs1.annotate(f'{corr1:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs2.annotate(f'{corr2:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs3.annotate(f'{corr3:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')

plt.tight_layout()


outfile = r'...\Fig_7\CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')





##South Atlantic 
fig, (axs1, axs2, axs3) = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

## Surface / Bottom Total Annual MHW Days ##
Td_SA = MHW_td_SA_MODEL/BMHW_td_SA_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_SA) & ~np.isnan(Td_SA)
MLT_Bottom_SA_clean = MLT_Bottom_SA[valid_indices]
Td_SA_clean = Td_SA[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_SA_clean,
    Td_SA_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

cmap=plt.cm.hot_r
vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
cs1 = axs1.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs1.set_xlabel(r'MLT / Bathy', fontsize=14)
axs1.set_ylabel(r'Total Annual SMHW / BMHW Days', fontsize=14)
axs1.set_title(r'South Atlantic (SA)', fontsize=14)
axs1.xaxis.set_minor_locator(AutoMinorLocator())
axs1.yaxis.set_minor_locator(AutoMinorLocator())
axs1.minorticks_on()
axs1.grid(which='both', linestyle='-', linewidth=0.5)



## BMHW/SMHW Max Intensity ##
MaxInt_SA = BMHW_max_SA_MODEL/MHW_max_SA_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_SA) & ~np.isnan(MaxInt_SA)
MLT_Bottom_SA_clean = MLT_Bottom_SA[valid_indices]
MaxInt_SA_clean = MaxInt_SA[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_SA_clean,
    MaxInt_SA_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im2 = axs2.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs2.set_xlabel(r'MLT / Bathy', fontsize=14)
axs2.set_ylabel(r'BMHW / SMHW Max Intensity', fontsize=14)
axs2.set_title('South Atlantic (SA)', fontsize=14)
axs2.xaxis.set_minor_locator(AutoMinorLocator())
axs2.yaxis.set_minor_locator(AutoMinorLocator())
axs2.minorticks_on()
axs2.grid(which='both', linestyle='-', linewidth=0.5)


## BMHW/SMHW Cum Intensity ##
CumInt_SA = BMHW_cum_SA_MODEL/MHW_cum_SA_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_SA) & ~np.isnan(CumInt_SA)
MLT_Bottom_SA_clean = MLT_Bottom_SA[valid_indices]
CumInt_SA_clean = CumInt_SA[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_SA_clean,
    CumInt_SA_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im3 = axs3.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs3.set_xlabel(r'MLT / Bathy', fontsize=14)
axs3.set_ylabel(r'BMHW / SMHW Cum Intensity', fontsize=14)
axs3.set_title('South Atlantic (SA)', fontsize=14)
axs3.xaxis.set_minor_locator(AutoMinorLocator())
axs3.yaxis.set_minor_locator(AutoMinorLocator())
axs3.minorticks_on()
axs3.grid(which='both', linestyle='-', linewidth=0.5)


# Calculate Spearman correlation for each subplot
corr1, _ = spearmanr(MLT_Bottom_SA_clean, Td_SA_clean)
corr2, _ = spearmanr(MLT_Bottom_SA_clean, MaxInt_SA_clean)
corr3, _ = spearmanr(MLT_Bottom_SA_clean, CumInt_SA_clean)

# Annotate each subplot with the Spearman correlation value
axs1.annotate(f'{corr1:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs2.annotate(f'{corr2:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs3.annotate(f'{corr3:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')

plt.tight_layout()


outfile = r'...\Fig_7\SA.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




##Levantine-Balearic 
fig, (axs1, axs2, axs3) = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

## Surface / Bottom Total Annual MHW Days ##
Td_BAL = MHW_td_BAL_MODEL/BMHW_td_BAL_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_BAL) & ~np.isnan(Td_BAL)
MLT_Bottom_BAL_clean = MLT_Bottom_BAL[valid_indices]
Td_BAL_clean = Td_BAL[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_BAL_clean,
    Td_BAL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

cmap=plt.cm.hot_r
vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
cs1 = axs1.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs1.set_xlabel(r'MLT / Bathy', fontsize=14)
axs1.set_ylabel(r'Total Annual SMHW / BMHW Days', fontsize=14)
axs1.set_title(r'Levantine-Balearic (BAL)', fontsize=14)
axs1.xaxis.set_minor_locator(AutoMinorLocator())
axs1.yaxis.set_minor_locator(AutoMinorLocator())
axs1.minorticks_on()
axs1.grid(which='both', linestyle='-', linewidth=0.5)



## BMHW/SMHW Max Intensity ##
MaxInt_BAL = BMHW_max_BAL_MODEL/MHW_max_BAL_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_BAL) & ~np.isnan(MaxInt_BAL)
MLT_Bottom_BAL_clean = MLT_Bottom_BAL[valid_indices]
MaxInt_BAL_clean = MaxInt_BAL[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_BAL_clean,
    MaxInt_BAL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im2 = axs2.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs2.set_xlabel(r'MLT / Bathy', fontsize=14)
axs2.set_ylabel(r'BMHW / SMHW Max Intensity', fontsize=14)
axs2.set_title('Levantine-Balearic (BAL)', fontsize=14)
axs2.xaxis.set_minor_locator(AutoMinorLocator())
axs2.yaxis.set_minor_locator(AutoMinorLocator())
axs2.minorticks_on()
axs2.grid(which='both', linestyle='-', linewidth=0.5)


## BMHW/SMHW Cum Intensity ##
CumInt_BAL = BMHW_cum_BAL_MODEL/MHW_cum_BAL_MODEL

#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_BAL) & ~np.isnan(CumInt_BAL)
MLT_Bottom_BAL_clean = MLT_Bottom_BAL[valid_indices]
CumInt_BAL_clean = CumInt_BAL[valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_BAL_clean,
    CumInt_BAL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

# Create the plot in axs1
im3 = axs3.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs3.set_xlabel(r'MLT / Bathy', fontsize=14)
axs3.set_ylabel(r'BMHW / SMHW Cum Intensity', fontsize=14)
axs3.set_title('Levantine-Balearic (BAL)', fontsize=14)
axs3.xaxis.set_minor_locator(AutoMinorLocator())
axs3.yaxis.set_minor_locator(AutoMinorLocator())
axs3.minorticks_on()
axs3.grid(which='both', linestyle='-', linewidth=0.5)


# Calculate Spearman correlation for each subplot
corr1, _ = spearmanr(MLT_Bottom_BAL_clean, Td_BAL_clean)
corr2, _ = spearmanr(MLT_Bottom_BAL_clean, MaxInt_BAL_clean)
corr3, _ = spearmanr(MLT_Bottom_BAL_clean, CumInt_BAL_clean)

# Annotate each subplot with the Spearman correlation value
axs1.annotate(f'{corr1:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs2.annotate(f'{corr2:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')
axs3.annotate(f'{corr3:.2f}', xy=(0.75, 0.05), xycoords='axes fraction', fontsize=14, color='black')

plt.tight_layout()


outfile = r'...\Fig_7\BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')
