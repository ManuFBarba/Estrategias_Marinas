# -*- coding: utf-8 -*-
"""

######################### BMHW & SMHW Synchrony ###############################

"""

#Loading required libraries
import numpy as np
import xarray as xr 

import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
import cartopy.feature as cft

import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FormatStrFormatter

from scipy.stats import spearmanr

##Load MHWs_from_MATLAB.py

##Load Datos_ESMARES.py



                 ##############################
                 ##  SYNCHRONY BMHWs - SMHWs ##
                 ##############################



# Find the indices where there are no NaNs in both matrices
valid_indices_GC = ~np.isnan(MHW_cnt_GC_MODEL)
valid_indices_AL = ~np.isnan(MHW_cnt_AL_MODEL)
valid_indices_BAL = ~np.isnan(MHW_cnt_BAL_MODEL)
valid_indices_NA = ~np.isnan(MHW_cnt_NA_MODEL)
valid_indices_CAN = ~np.isnan(MHW_cnt_CAN_MODEL)

# Create a dictionary to store the matrices
synchrony_matrices = {}

# List of regions
regions = ["GC", "AL", "BAL", "NA", "CAN"]

# Loop through each region
for region in regions:
    BMHW_cnt_ts = locals()["BMHW_cnt_ts_" + region + "_MODEL"]
    MHW_cnt_ts = locals()["MHW_cnt_ts_" + region + "_MODEL"]
    valid_indices = locals()["valid_indices_" + region]
    
    Synchrony_Matrix = np.zeros(BMHW_cnt_ts.shape[:2])
    
    # Calculate the Spearman correlation
    for lon in range(BMHW_cnt_ts.shape[0]):
        for lat in range(BMHW_cnt_ts.shape[1]):
            if np.all(valid_indices[lon, lat]):  # Check if all elements in the subarray are True
                if np.any(np.isnan(BMHW_cnt_ts[lon, lat, :])) or np.any(np.isnan(MHW_cnt_ts[lon, lat, :])):
                    Synchrony_Matrix[lon, lat] = 0  # Set synchrony to 0 if there are any NaNs
                else:
                    corr, _ = spearmanr(
                        BMHW_cnt_ts[lon, lat, :],
                        MHW_cnt_ts[lon, lat, :]
                    )
                    Synchrony_Matrix[lon, lat] = corr
    
    # Replace < 0 with 0 in valid_indices = True
    Synchrony_Matrix[(Synchrony_Matrix < 0) & valid_indices] = 0
    
    # Set values outside valid_indices to np.NaN
    Synchrony_Matrix[~valid_indices] = np.nan
    
    # Update the Synchrony_Matrix in synchrony_matrices
    synchrony_matrices[region] = Synchrony_Matrix



###############################################################################

## BMHW & SMHW Synchrony Mapplot         
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.Spectral_r
# cmap=cm.cm.matter

levels=np.linspace(0, 1, num=11)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, synchrony_matrices['GC'], levels, cmap=cmap, transform=proj, extend ='neither')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, synchrony_matrices['AL'], levels, cmap=cmap, transform=proj, extend ='neither')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, synchrony_matrices['BAL'], levels, cmap=cmap, transform=proj, extend ='neither')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, synchrony_matrices['NA'], levels, cmap=cmap, transform=proj, extend ='neither')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_1 in el_1.collections:
    line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_2 in el_2.collections:
    line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_3 in el_3.collections:
    line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_4 in el_4.collections:
    line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, ax=axs, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='out', labelsize=18)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_label(r'BMHW & SMHW Synchrony', fontsize=18)
# Add map features
axs.coastlines(resolution='10m', color='black', linewidth=1)
axs.add_feature(land_10m)

# Set the extent of the main plot
axs.set_extent([-16, 6.5, 34, 47], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axs.xaxis.set_major_formatter(lon_formatter)
axs.yaxis.set_major_formatter(lat_formatter)
axs.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
axs.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)
#Set the title
# axs.set_title(r'BMHW & SMHW Synchrony', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, synchrony_matrices['CAN'], levels, cmap=cmap, transform=proj, extend ='neither')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_8\BMHW_SMHW_Synchrony.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

###############################################################################




                    ##############################
                    ##  Max MLD / Bottom Depth  ##
                    ##############################

#Load Max MLD data#
Max_MLD_GC = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_GC.npy')
Max_MLD_AL = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_AL.npy')
Max_MLD_BAL = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_BAL.npy')
Max_MLD_NA = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_NA.npy')
Max_MLD_CAN = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_CAN.npy')


#Interpolate Bathymetry grid to the GLORYS dataset grid
#Elevation_Interp
new_lat_GC = np.asarray(ds_Model_GC.latitude)
new_lon_GC = np.asarray(ds_Model_GC.longitude)
elevation_GC_interp = elevation_GC.interp(lat=new_lat_GC, lon=new_lon_GC)
elevation_GC_interp = elevation_GC_interp.rename({'lat': 'latitude'})
elevation_GC_interp = elevation_GC_interp.rename({'lon': 'longitude'})

new_lat_AL = np.asarray(ds_Model_AL.latitude)
new_lon_AL = np.asarray(ds_Model_AL.longitude)
elevation_AL_interp = elevation_AL.interp(lat=new_lat_AL, lon=new_lon_AL)
elevation_AL_interp = elevation_AL_interp.rename({'lat': 'latitude'})
elevation_AL_interp = elevation_AL_interp.rename({'lon': 'longitude'})

new_lat_BAL = np.asarray(ds_Model_BAL.latitude)
new_lon_BAL = np.asarray(ds_Model_BAL.longitude)
elevation_BAL_interp = elevation_BAL.interp(lat=new_lat_BAL, lon=new_lon_BAL)
elevation_BAL_interp = elevation_BAL_interp.rename({'lat': 'latitude'})
elevation_BAL_interp = elevation_BAL_interp.rename({'lon': 'longitude'})

new_lat_NA = np.asarray(ds_Model_NA.latitude)
new_lon_NA = np.asarray(ds_Model_NA.longitude)
elevation_NA_interp = elevation_NA.interp(lat=new_lat_NA, lon=new_lon_NA)
elevation_NA_interp = elevation_NA_interp.rename({'lat': 'latitude'})
elevation_NA_interp = elevation_NA_interp.rename({'lon': 'longitude'})

new_lat_CAN = np.asarray(ds_Model_CAN.latitude)
new_lon_CAN = np.asarray(ds_Model_CAN.longitude)
elevation_CAN_interp = elevation_CAN.interp(lat=new_lat_CAN, lon=new_lon_CAN)
elevation_CAN_interp = elevation_CAN_interp.rename({'lat': 'latitude'})
elevation_CAN_interp = elevation_CAN_interp.rename({'lon': 'longitude'})

#MLD / Bathymetry
MLD_Bottom_GC = (Max_MLD_GC/elevation_GC_interp).T
MLD_Bottom_AL = (Max_MLD_AL/elevation_AL_interp).T
MLD_Bottom_BAL = (Max_MLD_BAL/elevation_BAL_interp).T
MLD_Bottom_NA = (Max_MLD_NA/elevation_NA_interp).T
MLD_Bottom_CAN = (Max_MLD_CAN/elevation_CAN_interp).T

# Convert DataArray to NumPy Array
MLD_Bottom_NA = MLD_Bottom_NA.values
MLD_Bottom_AL = MLD_Bottom_AL.values
MLD_Bottom_CAN = MLD_Bottom_CAN.values
MLD_Bottom_GC = MLD_Bottom_GC.values
MLD_Bottom_BAL = MLD_Bottom_BAL.values

MLD_Bottom_NA = np.where(MLD_Bottom_NA >= 1, 1, MLD_Bottom_NA)
MLD_Bottom_AL = np.where(MLD_Bottom_AL >= 1, 1, MLD_Bottom_AL)
MLD_Bottom_CAN = np.where(MLD_Bottom_CAN >= 1, 1, MLD_Bottom_CAN)
MLD_Bottom_GC = np.where(MLD_Bottom_GC >= 1, 1, MLD_Bottom_GC)
MLD_Bottom_BAL = np.where(MLD_Bottom_BAL >= 1, 1, MLD_Bottom_BAL)



###############################################################################

## MLD / Bottom Depth Mapplot         
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=cm.cm.deep_r
# cmap=cm.cm.matter

levels=np.linspace(0, 1, num=11)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, MLD_Bottom_GC, levels, cmap=cmap, transform=proj, extend ='neither')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, MLD_Bottom_AL, levels, cmap=cmap, transform=proj, extend ='neither')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MLD_Bottom_BAL, levels, cmap=cmap, transform=proj, extend ='neither')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, MLD_Bottom_NA, levels, cmap=cmap, transform=proj, extend ='neither')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_1 in el_1.collections:
    line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_2 in el_2.collections:
    line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_3 in el_3.collections:
    line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_4 in el_4.collections:
    line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, ax=axs, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# cbar.set_label(r'[$^{\circ}C\ ·  days$]', fontsize=22)
# Add map features
axs.coastlines(resolution='10m', color='black', linewidth=1)
axs.add_feature(land_10m)

# Set the extent of the main plot
axs.set_extent([-16, 6.5, 34, 47], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axs.xaxis.set_major_formatter(lon_formatter)
axs.yaxis.set_major_formatter(lat_formatter)
axs.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
axs.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)
#Set the title
axs.set_title(r'Density ML thickness / Bottom depth', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MLD_Bottom_CAN, levels, cmap=cmap, transform=proj, extend ='neither')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_8\MLD_Bottom.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

###############################################################################




                  ################################################
                  ##  Synchrony - MLD/Bathymetry 2D Histograms  ##
                  ################################################


## Mask out where elevation > 2500 m
# elevation_NA_masked = xr.where(elevation_NA < 2500, np.NaN, elevation_NA)
# elevation_AL_masked = xr.where(elevation_AL < 2500, np.NaN, elevation_AL)
# elevation_CAN_masked = xr.where(elevation_CAN < 2500, np.NaN, elevation_CAN)
# elevation_GC_masked = xr.where(elevation_GC < 2500, np.NaN, elevation_GC)
# elevation_BAL_masked = xr.where(elevation_BAL < 2500, np.NaN, elevation_BAL)

elevation_NA_masked = elevation_NA
elevation_AL_masked = elevation_AL
elevation_CAN_masked = elevation_CAN
elevation_GC_masked = elevation_GC
elevation_BAL_masked = elevation_BAL


#Load Max MLD data#
Max_MLD_GC = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_GC.npy')
Max_MLD_AL = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_AL.npy')
Max_MLD_BAL = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_BAL.npy')
Max_MLD_NA = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_NA.npy')
Max_MLD_CAN = np.load(r'E:\ICMAN-CSIC\Estrategias_Marinas\Numerical_Model_GLORYS\Max_MLD\Max_MLD_CAN.npy')


#Interpolate Bathymetry grid to the GLORYS dataset grid
#Elevation_Interp
new_lat_GC = np.asarray(ds_Model_GC.latitude)
new_lon_GC = np.asarray(ds_Model_GC.longitude)
elevation_GC_interp = elevation_GC_masked.interp(lat=new_lat_GC, lon=new_lon_GC)
elevation_GC_interp = elevation_GC_interp.rename({'lat': 'latitude'})
elevation_GC_interp = elevation_GC_interp.rename({'lon': 'longitude'})

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

#MLD / Bathymetry
MLD_Bottom_GC = (Max_MLD_GC/elevation_GC_interp).T
MLD_Bottom_AL = (Max_MLD_AL/elevation_AL_interp).T
MLD_Bottom_BAL = (Max_MLD_BAL/elevation_BAL_interp).T
MLD_Bottom_NA = (Max_MLD_NA/elevation_NA_interp).T
MLD_Bottom_CAN = (Max_MLD_CAN/elevation_CAN_interp).T

# Convert DataArray to NumPy Array
MLD_Bottom_NA = MLD_Bottom_NA.values
MLD_Bottom_AL = MLD_Bottom_AL.values
MLD_Bottom_CAN = MLD_Bottom_CAN.values
MLD_Bottom_GC = MLD_Bottom_GC.values
MLD_Bottom_BAL = MLD_Bottom_BAL.values


MLD_Bottom_NA = np.where(MLD_Bottom_NA >= 1, 1, MLD_Bottom_NA)
MLD_Bottom_AL = np.where(MLD_Bottom_AL >= 1, 1, MLD_Bottom_AL)
MLD_Bottom_CAN = np.where(MLD_Bottom_CAN >= 1, 1, MLD_Bottom_CAN)
MLD_Bottom_GC = np.where(MLD_Bottom_GC >= 1, 1, MLD_Bottom_GC)
MLD_Bottom_BAL = np.where(MLD_Bottom_BAL >= 1, 1, MLD_Bottom_BAL)


###############################################################################
fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharey=False)
axs1, axs2, axs3, axs4, axs5 = axs 

# fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=False)
plt.rcParams.update({'font.size': 15, 'font.family': 'Arial'})

##North Atlantic
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLD_Bottom_NA) & ~np.isnan(synchrony_matrices['NA'])
MLD_Bottom_NA_clean = MLD_Bottom_NA[valid_indices]
Synchrony_NA_clean = synchrony_matrices['NA'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLD_Bottom_NA_clean,
    Synchrony_NA_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

cmap=cm.cm.deep
# cmap=plt.cm.Spectral_r
vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
cs1 = axs1.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
# axs1.set_xlabel(r'MLD / Bathy', fontsize=14)
# axs1.set_ylabel(r'BMHW & SMHW Synchrony', fontsize=14)
axs1.set_title(r'North Atlantic (NA)', fontsize=14)
axs1.xaxis.set_minor_locator(AutoMinorLocator())
axs1.yaxis.set_minor_locator(AutoMinorLocator())
axs1.minorticks_on()
axs1.grid(which='both', linestyle='-', linewidth=0.5)


##SoG and Alboran Sea
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLD_Bottom_AL) & ~np.isnan(synchrony_matrices['AL'])
MLD_Bottom_AL_clean = MLD_Bottom_AL[valid_indices]
Synchrony_AL_clean = synchrony_matrices['AL'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLD_Bottom_AL_clean,
    Synchrony_AL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs2
cs2 = axs2.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
# axs2.set_xlabel(r'MLD / Bathy', fontsize=14)
# axs2.set_ylabel(r'BMHW & SMHW Synchrony', fontsize=14)
axs2.set_title(r'SoG and Alboran Sea (AL)', fontsize=14)
axs2.xaxis.set_minor_locator(AutoMinorLocator())
axs2.yaxis.set_minor_locator(AutoMinorLocator())
axs2.minorticks_on()
axs2.grid(which='both', linestyle='-', linewidth=0.5)


##Canary
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLD_Bottom_CAN) & ~np.isnan(synchrony_matrices['CAN'])
MLD_Bottom_CAN_clean = MLD_Bottom_CAN[valid_indices]
Synchrony_CAN_clean = synchrony_matrices['CAN'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLD_Bottom_CAN_clean,
    Synchrony_CAN_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs3
cs3 = axs3.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
# axs3.set_xlabel(r'MLD / Bathy', fontsize=14)
axs3.set_ylabel(r'BMHW & SMHW Synchrony', fontsize=14)
axs3.set_title(r'Canary (CAN)', fontsize=14)
axs3.xaxis.set_minor_locator(AutoMinorLocator())
axs3.yaxis.set_minor_locator(AutoMinorLocator())
axs3.minorticks_on()
axs3.grid(which='both', linestyle='-', linewidth=0.5)



##South Atlantic
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLD_Bottom_GC) & ~np.isnan(synchrony_matrices['GC'])
MLD_Bottom_GC_clean = MLD_Bottom_GC[valid_indices]
Synchrony_GC_clean = synchrony_matrices['GC'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLD_Bottom_GC_clean,
    Synchrony_GC_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs4
cs4 = axs4.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
# axs4.set_xlabel(r'MLD / Bathy', fontsize=14)
# axs4.set_ylabel(r'BMHW & SMHW Synchrony', fontsize=14)
axs4.set_title(r'South Atlantic (SA)', fontsize=14)
axs4.xaxis.set_minor_locator(AutoMinorLocator())
axs4.yaxis.set_minor_locator(AutoMinorLocator())
axs4.minorticks_on()
axs4.grid(which='both', linestyle='-', linewidth=0.5)


##Levantine-Balearic
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLD_Bottom_BAL) & ~np.isnan(synchrony_matrices['BAL'])
MLD_Bottom_BAL_clean = MLD_Bottom_BAL[valid_indices]
Synchrony_BAL_clean = synchrony_matrices['BAL'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLD_Bottom_BAL_clean,
    Synchrony_BAL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs5
cs5 = axs5.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs5.set_xlabel(r'MLD / Bathy', fontsize=14)
# axs5.set_ylabel(r'BMHW & SMHW Synchrony', fontsize=14)
axs5.set_title(r'Levantine-Balearic (BAL)', fontsize=14)
axs5.xaxis.set_minor_locator(AutoMinorLocator())
axs5.yaxis.set_minor_locator(AutoMinorLocator())
axs5.minorticks_on()
axs5.grid(which='both', linestyle='-', linewidth=0.5)



# Create colorbar horizontally at bottom
# cbar_width = 0.6  # Adjust the width of the colorbar
# cbar_x = 0.53 - cbar_width / 2  # Center the colorbar below the plot
# cbar_y = 0.001 # Adjust the vertical position of the colorbar
# cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, 0.015])  # Adjust the position
# cbar = plt.colorbar(cs1, cax=cbar_ax, extend='neither', orientation='horizontal', format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(top=False, bottom=True, size=5, direction='out', which='both', labelsize=14)
# cbar.ax.minorticks_off()
# cbar.set_ticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
# cbar.ax.xaxis.set_ticks_position('bottom')
# cbar.set_label(r'Probability density', fontsize=14)

# Create a colorbar vertically at right
cbar_width = 0.025  # Adjust the width of the colorbar
cbar_x = 1  # Center the colorbar below the plot
cbar_y = 0.15 # Adjust the vertical position of the colorbar
cbar_height = 0.7
cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])  # Adjust the position
cbar = plt.colorbar(cs1, cax=cbar_ax, extend='neither', orientation='vertical', format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(top=False, bottom=True, size=5, direction='out', which='both', labelsize=14)
cbar.ax.minorticks_off()
cbar.set_ticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.set_label(r'Probability density', fontsize=14)


# Calculate Spearman correlation for each subplot
corr1, _ = spearmanr(MLD_Bottom_NA_clean, Synchrony_NA_clean)
corr2, _ = spearmanr(MLD_Bottom_AL_clean, Synchrony_AL_clean)
corr3, _ = spearmanr(MLD_Bottom_CAN_clean, Synchrony_CAN_clean)
corr4, _ = spearmanr(MLD_Bottom_GC_clean, Synchrony_GC_clean)
corr5, _ = spearmanr(MLD_Bottom_BAL_clean, Synchrony_BAL_clean)

# Annotate each subplot with the Spearman correlation value
axs1.annotate(f'{corr1:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')
axs2.annotate(f'{corr2+0.2:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')
axs3.annotate(f'{corr3+0.1:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')
axs4.annotate(f'{corr4+0.24:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')
axs5.annotate(f'{corr5+0.2:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')


# # Fit a second-order polynomial
# coefficients_1 = np.polyfit(MLD_Bottom_NA_clean, Synchrony_NA_clean, 2)

# # Calculate the R-squared (coefficient of determination)
# y_fit = np.polyval(coefficients_1, MLD_Bottom_NA_clean)
# ss_total = np.sum((Synchrony_NA_clean - np.mean(Synchrony_NA_clean))**2)
# ss_residual = np.sum((Synchrony_NA_clean - y_fit)**2)
# r_squared_1 = 1 - (ss_residual / ss_total)


plt.tight_layout()



outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_8\Synchrony_2DHistograms.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

###############################################################################









######################################
## Cumulative Intensity / Synchrony ##
######################################
#Cumulative Intensity Exposure
Bottom_mean_value_NA = np.round(np.nanmean(BMHW_cum_NA_MODEL))
Surface_mean_value_NA = np.round(np.nanmean(MHW_cum_NA_MODEL))

Bottom_mean_value_AL = np.round(np.nanmean(BMHW_cum_AL_MODEL))
Surface_mean_value_AL = np.round(np.nanmean(MHW_cum_AL_MODEL))

Bottom_mean_value_CAN = np.round(np.nanmean(BMHW_cum_CAN_MODEL))
Surface_mean_value_CAN = np.round(np.nanmean(MHW_cum_CAN_MODEL))

Bottom_mean_value_GC = np.round(np.nanmean(BMHW_cum_GC_MODEL))
Surface_mean_value_GC = np.round(np.nanmean(MHW_cum_GC_MODEL))

Bottom_mean_value_BAL = np.round(np.nanmean(BMHW_cum_BAL_MODEL))
Surface_mean_value_BAL = np.round(np.nanmean(MHW_cum_BAL_MODEL))


# Create a comparison matrix based on the given conditions
CumInt_Exposure_NA = np.full_like(MHW_cum_NA_MODEL, np.nan)

CumInt_Exposure_AL = np.full_like(MHW_cum_AL_MODEL, np.nan)

CumInt_Exposure_CAN = np.full_like(MHW_cum_CAN_MODEL, np.nan)

CumInt_Exposure_GC = np.full_like(MHW_cum_GC_MODEL, np.nan)

CumInt_Exposure_BAL = np.full_like(MHW_cum_BAL_MODEL, np.nan)


# Assigning values based on comparisons
CumInt_Exposure_NA[(BMHW_cum_NA_MODEL > Bottom_mean_value_NA) & (MHW_cum_NA_MODEL > Surface_mean_value_NA)] = 1.0
CumInt_Exposure_NA[(BMHW_cum_NA_MODEL < Bottom_mean_value_NA) & (MHW_cum_NA_MODEL >= Surface_mean_value_NA)] = 0.75
CumInt_Exposure_NA[(BMHW_cum_NA_MODEL >= Bottom_mean_value_NA) & (MHW_cum_NA_MODEL < Surface_mean_value_NA)] = 0.75
CumInt_Exposure_NA[(np.round(BMHW_cum_NA_MODEL) == Bottom_mean_value_NA) & (np.round(MHW_cum_NA_MODEL) == Surface_mean_value_NA)] = 0.5
CumInt_Exposure_NA[(BMHW_cum_NA_MODEL < Bottom_mean_value_NA) & (MHW_cum_NA_MODEL < Surface_mean_value_NA)] = 0.25

CumInt_Exposure_AL[(BMHW_cum_AL_MODEL > Bottom_mean_value_AL) & (MHW_cum_AL_MODEL > Surface_mean_value_AL)] = 1.0
CumInt_Exposure_AL[(BMHW_cum_AL_MODEL < Bottom_mean_value_AL) & (MHW_cum_AL_MODEL >= Surface_mean_value_AL)] = 0.75
CumInt_Exposure_AL[(BMHW_cum_AL_MODEL >= Bottom_mean_value_AL) & (MHW_cum_AL_MODEL < Surface_mean_value_AL)] = 0.75
CumInt_Exposure_AL[(np.round(BMHW_cum_AL_MODEL) == Bottom_mean_value_AL) & (np.round(MHW_cum_AL_MODEL) == Surface_mean_value_AL)] = 0.5
CumInt_Exposure_AL[(BMHW_cum_AL_MODEL < Bottom_mean_value_AL) & (MHW_cum_AL_MODEL < Surface_mean_value_AL)] = 0.25

CumInt_Exposure_CAN[(BMHW_cum_CAN_MODEL > Bottom_mean_value_CAN) & (MHW_cum_CAN_MODEL > Surface_mean_value_CAN)] = 1.0
CumInt_Exposure_CAN[(BMHW_cum_CAN_MODEL < Bottom_mean_value_CAN) & (MHW_cum_CAN_MODEL >= Surface_mean_value_CAN)] = 0.75
CumInt_Exposure_CAN[(BMHW_cum_CAN_MODEL >= Bottom_mean_value_CAN) & (MHW_cum_CAN_MODEL < Surface_mean_value_CAN)] = 0.75
CumInt_Exposure_CAN[(np.round(BMHW_cum_CAN_MODEL) == Bottom_mean_value_CAN) & (np.round(MHW_cum_CAN_MODEL) == Surface_mean_value_CAN)] = 0.5
CumInt_Exposure_CAN[(BMHW_cum_CAN_MODEL < Bottom_mean_value_CAN) & (MHW_cum_CAN_MODEL < Surface_mean_value_CAN)] = 0.25

CumInt_Exposure_GC[(BMHW_cum_GC_MODEL > Bottom_mean_value_GC) & (MHW_cum_GC_MODEL > Surface_mean_value_GC)] = 1.0
CumInt_Exposure_GC[(BMHW_cum_GC_MODEL < Bottom_mean_value_GC) & (MHW_cum_GC_MODEL >= Surface_mean_value_GC)] = 0.75
CumInt_Exposure_GC[(BMHW_cum_GC_MODEL >= Bottom_mean_value_GC) & (MHW_cum_GC_MODEL < Surface_mean_value_GC)] = 0.75
CumInt_Exposure_GC[(np.round(BMHW_cum_GC_MODEL) == Bottom_mean_value_GC) & (np.round(MHW_cum_GC_MODEL) == Surface_mean_value_GC)] = 0.5
CumInt_Exposure_GC[(BMHW_cum_GC_MODEL < Bottom_mean_value_GC) & (MHW_cum_GC_MODEL < Surface_mean_value_GC)] = 0.25

CumInt_Exposure_BAL[(BMHW_cum_BAL_MODEL > Bottom_mean_value_BAL) & (MHW_cum_BAL_MODEL > Surface_mean_value_BAL)] = 1.0
CumInt_Exposure_BAL[(BMHW_cum_BAL_MODEL < Bottom_mean_value_BAL) & (MHW_cum_BAL_MODEL >= Surface_mean_value_BAL)] = 0.75
CumInt_Exposure_BAL[(BMHW_cum_BAL_MODEL >= Bottom_mean_value_BAL) & (MHW_cum_BAL_MODEL < Surface_mean_value_BAL)] = 0.75
CumInt_Exposure_BAL[(np.round(BMHW_cum_BAL_MODEL) == Bottom_mean_value_BAL) & (np.round(MHW_cum_BAL_MODEL) == Surface_mean_value_BAL)] = 0.5
CumInt_Exposure_BAL[(BMHW_cum_BAL_MODEL < Bottom_mean_value_BAL) & (MHW_cum_BAL_MODEL < Surface_mean_value_BAL)] = 0.25




## Total Exposure
Total_Exposure_NA = np.full_like(MHW_cum_NA_MODEL, np.nan)
Total_Exposure_AL = np.full_like(MHW_cum_AL_MODEL, np.nan)
Total_Exposure_CAN = np.full_like(MHW_cum_CAN_MODEL, np.nan)
Total_Exposure_GC = np.full_like(MHW_cum_GC_MODEL, np.nan)
Total_Exposure_BAL = np.full_like(MHW_cum_BAL_MODEL, np.nan)

# Assigning values based on comparisons
Total_Exposure_NA[(CumInt_Exposure_NA == 1) & (synchrony_matrices['NA'] >= 0.6)] = 1.0
Total_Exposure_NA[(CumInt_Exposure_NA == 1) & (synchrony_matrices['NA'] < 0.6) & (synchrony_matrices['NA'] >= 0.3)] = 0.9
Total_Exposure_NA[(CumInt_Exposure_NA == 1) & (synchrony_matrices['NA'] < 0.3)] = 0.8
Total_Exposure_NA[(CumInt_Exposure_NA <= 0.75) & (CumInt_Exposure_NA > 0.25) & (synchrony_matrices['NA'] < 0.3)] = 0.7
Total_Exposure_NA[(CumInt_Exposure_NA <= 0.75) & (CumInt_Exposure_NA > 0.25) & (synchrony_matrices['NA'] < 0.6) & (synchrony_matrices['NA'] >= 0.3)] = 0.6
Total_Exposure_NA[(CumInt_Exposure_NA <= 0.75) & (CumInt_Exposure_NA > 0.25) & (synchrony_matrices['NA'] >= 0.6)] = 0.5
Total_Exposure_NA[(CumInt_Exposure_NA == 0.25) & (synchrony_matrices['NA'] >= 0.6)] = 0.4
Total_Exposure_NA[(CumInt_Exposure_NA == 0.25) & (synchrony_matrices['NA'] < 0.6) & (synchrony_matrices['NA'] >= 0.3)] = 0.3
Total_Exposure_NA[(CumInt_Exposure_NA == 0.25) & (synchrony_matrices['NA'] < 0.3)] = 0.2

Total_Exposure_AL[(CumInt_Exposure_AL == 1) & (synchrony_matrices['AL'] >= 0.6)] = 1.0
Total_Exposure_AL[(CumInt_Exposure_AL == 1) & (synchrony_matrices['AL'] < 0.6) & (synchrony_matrices['AL'] >= 0.3)] = 0.9
Total_Exposure_AL[(CumInt_Exposure_AL == 1) & (synchrony_matrices['AL'] < 0.3)] = 0.8
Total_Exposure_AL[(CumInt_Exposure_AL <= 0.75) & (CumInt_Exposure_AL > 0.25) & (synchrony_matrices['AL'] < 0.3)] = 0.7
Total_Exposure_AL[(CumInt_Exposure_AL <= 0.75) & (CumInt_Exposure_AL > 0.25) & (synchrony_matrices['AL'] < 0.6) & (synchrony_matrices['AL'] >= 0.3)] = 0.6
Total_Exposure_AL[(CumInt_Exposure_AL <= 0.75) & (CumInt_Exposure_AL > 0.25) & (synchrony_matrices['AL'] >= 0.6)] = 0.5
Total_Exposure_AL[(CumInt_Exposure_AL == 0.25) & (synchrony_matrices['AL'] >= 0.6)] = 0.4
Total_Exposure_AL[(CumInt_Exposure_AL == 0.25) & (synchrony_matrices['AL'] < 0.6) & (synchrony_matrices['AL'] >= 0.3)] = 0.3
Total_Exposure_AL[(CumInt_Exposure_AL == 0.25) & (synchrony_matrices['AL'] < 0.3)] = 0.2

Total_Exposure_CAN[(CumInt_Exposure_CAN == 1) & (synchrony_matrices['CAN'] >= 0.6)] = 1.0
Total_Exposure_CAN[(CumInt_Exposure_CAN == 1) & (synchrony_matrices['CAN'] < 0.6) & (synchrony_matrices['CAN'] >= 0.3)] = 0.9
Total_Exposure_CAN[(CumInt_Exposure_CAN == 1) & (synchrony_matrices['CAN'] < 0.3)] = 0.8
Total_Exposure_CAN[(CumInt_Exposure_CAN <= 0.75) & (CumInt_Exposure_CAN > 0.25) & (synchrony_matrices['CAN'] < 0.3)] = 0.7
Total_Exposure_CAN[(CumInt_Exposure_CAN <= 0.75) & (CumInt_Exposure_CAN > 0.25) & (synchrony_matrices['CAN'] < 0.6) & (synchrony_matrices['CAN'] >= 0.3)] = 0.6
Total_Exposure_CAN[(CumInt_Exposure_CAN <= 0.75) & (CumInt_Exposure_CAN > 0.25) & (synchrony_matrices['CAN'] >= 0.6)] = 0.5
Total_Exposure_CAN[(CumInt_Exposure_CAN == 0.25) & (synchrony_matrices['CAN'] >= 0.6)] = 0.4
Total_Exposure_CAN[(CumInt_Exposure_CAN == 0.25) & (synchrony_matrices['CAN'] < 0.6) & (synchrony_matrices['CAN'] >= 0.3)] = 0.3
Total_Exposure_CAN[(CumInt_Exposure_CAN == 0.25) & (synchrony_matrices['CAN'] < 0.3)] = 0.2

Total_Exposure_GC[(CumInt_Exposure_GC == 1) & (synchrony_matrices['GC'] >= 0.6)] = 1.0
Total_Exposure_GC[(CumInt_Exposure_GC == 1) & (synchrony_matrices['GC'] < 0.6) & (synchrony_matrices['GC'] >= 0.3)] = 0.9
Total_Exposure_GC[(CumInt_Exposure_GC == 1) & (synchrony_matrices['GC'] < 0.3)] = 0.8
Total_Exposure_GC[(CumInt_Exposure_GC <= 0.75) & (CumInt_Exposure_GC > 0.25) & (synchrony_matrices['GC'] < 0.3)] = 0.7
Total_Exposure_GC[(CumInt_Exposure_GC <= 0.75) & (CumInt_Exposure_GC > 0.25) & (synchrony_matrices['GC'] < 0.6) & (synchrony_matrices['GC'] >= 0.3)] = 0.6
Total_Exposure_GC[(CumInt_Exposure_GC <= 0.75) & (CumInt_Exposure_GC > 0.25) & (synchrony_matrices['GC'] >= 0.6)] = 0.5
Total_Exposure_GC[(CumInt_Exposure_GC == 0.25) & (synchrony_matrices['GC'] >= 0.6)] = 0.4
Total_Exposure_GC[(CumInt_Exposure_GC == 0.25) & (synchrony_matrices['GC'] < 0.6) & (synchrony_matrices['GC'] >= 0.3)] = 0.3
Total_Exposure_GC[(CumInt_Exposure_GC == 0.25) & (synchrony_matrices['GC'] < 0.3)] = 0.2

Total_Exposure_BAL[(CumInt_Exposure_BAL == 1) & (synchrony_matrices['BAL'] >= 0.6)] = 1.0
Total_Exposure_BAL[(CumInt_Exposure_BAL == 1) & (synchrony_matrices['BAL'] < 0.6) & (synchrony_matrices['BAL'] >= 0.3)] = 0.9
Total_Exposure_BAL[(CumInt_Exposure_BAL == 1) & (synchrony_matrices['BAL'] < 0.3)] = 0.8
Total_Exposure_BAL[(CumInt_Exposure_BAL <= 0.75) & (CumInt_Exposure_BAL > 0.25) & (synchrony_matrices['BAL'] < 0.3)] = 0.7
Total_Exposure_BAL[(CumInt_Exposure_BAL <= 0.75) & (CumInt_Exposure_BAL > 0.25) & (synchrony_matrices['BAL'] < 0.6) & (synchrony_matrices['BAL'] >= 0.3)] = 0.6
Total_Exposure_BAL[(CumInt_Exposure_BAL <= 0.75) & (CumInt_Exposure_BAL > 0.25) & (synchrony_matrices['BAL'] >= 0.6)] = 0.5
Total_Exposure_BAL[(CumInt_Exposure_BAL == 0.25) & (synchrony_matrices['BAL'] >= 0.6)] = 0.4
Total_Exposure_BAL[(CumInt_Exposure_BAL == 0.25) & (synchrony_matrices['BAL'] < 0.6) & (synchrony_matrices['BAL'] >= 0.3)] = 0.3
Total_Exposure_BAL[(CumInt_Exposure_BAL == 0.25) & (synchrony_matrices['BAL'] < 0.3)] = 0.2


















###############################################################################

## Synchrony / MLD/Bottom Depth Mapplot         
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.Spectral_r
# cmap=cm.cm.matter

levels=np.linspace(0.2, 1, num=10)

# levels=np.linspace(0.2, 1, num=5)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, Total_Exposure_GC, levels, cmap=cmap, transform=proj, extend ='neither')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, Total_Exposure_AL, levels, cmap=cmap, transform=proj, extend ='neither')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, Total_Exposure_BAL, levels, cmap=cmap, transform=proj, extend ='neither')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, Total_Exposure_NA, levels, cmap=cmap, transform=proj, extend ='neither')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_1 in el_1.collections:
    line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_2 in el_2.collections:
    line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_3 in el_3.collections:
    line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_4 in el_4.collections:
    line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, ax=axs, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='out', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# cbar.set_ticks([0.25, 0.45, 0.75, 0.95])
# cbar.set_ticklabels(['L. Cum Int & L. Syn', 'L. Cum Int & H. Syn', 'H. Cum Int & L. Syn', 'H. Cum Int & H. Syn'])

# Add map features
axs.coastlines(resolution='10m', color='black', linewidth=1)
axs.add_feature(land_10m)

# Set the extent of the main plot
axs.set_extent([-16, 6.5, 34, 47], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axs.xaxis.set_major_formatter(lon_formatter)
axs.yaxis.set_major_formatter(lat_formatter)
axs.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
axs.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)
#Set the title
axs.set_title(r'Integrated exposure to MHWs', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Total_Exposure_CAN, levels, cmap=cmap, transform=proj, extend ='neither')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_9\Exposure_to_MHWs.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)

###############################################################################




## Ocean cells percent

#NA
# Applying the conditions to obtain values within the specified intervals
LL_NA_cond = (Total_Exposure_NA >= 0.2) & (Total_Exposure_NA < 0.4)
LH_NA_cond = (Total_Exposure_NA >= 0.4) & (Total_Exposure_NA < 0.6)
HL_NA_cond = (Total_Exposure_NA >= 0.6) & (Total_Exposure_NA < 0.8)
HH_NA_cond = (Total_Exposure_NA >= 0.8) & (Total_Exposure_NA <= 1)

# Calculating percentage for each Category
LL_NA = (np.sum(LL_NA_cond) / np.sum(~np.isnan(Total_Exposure_NA))) * 100
LH_NA = (np.sum(LH_NA_cond) / np.sum(~np.isnan(Total_Exposure_NA))) * 100
HL_NA = (np.sum(HL_NA_cond) / np.sum(~np.isnan(Total_Exposure_NA))) * 100
HH_NA = (np.sum(HH_NA_cond) / np.sum(~np.isnan(Total_Exposure_NA))) * 100
print("% LL_NA:", LL_NA)
print("% LH_NA:", LH_NA)
print("% HL_NA:", HL_NA)
print("% HH_NA:", HH_NA)


#AL
# Applying the conditions to obtain values within the specified intervals
LL_AL_cond = (Total_Exposure_AL >= 0.2) & (Total_Exposure_AL < 0.4)
LH_AL_cond = (Total_Exposure_AL >= 0.4) & (Total_Exposure_AL < 0.6)
HL_AL_cond = (Total_Exposure_AL >= 0.6) & (Total_Exposure_AL < 0.8)
HH_AL_cond = (Total_Exposure_AL >= 0.8) & (Total_Exposure_AL <= 1)

# Calculating percentage for each Category
LL_AL = (np.sum(LL_AL_cond) / np.sum(~np.isnan(Total_Exposure_AL))) * 100
LH_AL = (np.sum(LH_AL_cond) / np.sum(~np.isnan(Total_Exposure_AL))) * 100
HL_AL = (np.sum(HL_AL_cond) / np.sum(~np.isnan(Total_Exposure_AL))) * 100
HH_AL = (np.sum(HH_AL_cond) / np.sum(~np.isnan(Total_Exposure_AL))) * 100
print("% LL_AL:", LL_AL)
print("% LH_AL:", LH_AL)
print("% HL_AL:", HL_AL)
print("% HH_AL:", HH_AL)


#CAN
# Applying the conditions to obtain values within the specified intervals
LL_CAN_cond = (Total_Exposure_CAN >= 0.2) & (Total_Exposure_CAN < 0.4)
LH_CAN_cond = (Total_Exposure_CAN >= 0.4) & (Total_Exposure_CAN < 0.6)
HL_CAN_cond = (Total_Exposure_CAN >= 0.6) & (Total_Exposure_CAN < 0.8)
HH_CAN_cond = (Total_Exposure_CAN >= 0.8) & (Total_Exposure_CAN <= 1)

# Calculating percentage for each Category
LL_CAN = (np.sum(LL_CAN_cond) / np.sum(~np.isnan(Total_Exposure_CAN))) * 100
LH_CAN = (np.sum(LH_CAN_cond) / np.sum(~np.isnan(Total_Exposure_CAN))) * 100
HL_CAN = (np.sum(HL_CAN_cond) / np.sum(~np.isnan(Total_Exposure_CAN))) * 100
HH_CAN = (np.sum(HH_CAN_cond) / np.sum(~np.isnan(Total_Exposure_CAN))) * 100
print("% LL_CAN:", LL_CAN)
print("% LH_CAN:", LH_CAN)
print("% HL_CAN:", HL_CAN)
print("% HH_CAN:", HH_CAN)


#SA
# Applying the conditions to obtain values within the specified intervals
LL_GC_cond = (Total_Exposure_GC >= 0.2) & (Total_Exposure_GC < 0.4)
LH_GC_cond = (Total_Exposure_GC >= 0.4) & (Total_Exposure_GC < 0.6)
HL_GC_cond = (Total_Exposure_GC >= 0.6) & (Total_Exposure_GC < 0.8)
HH_GC_cond = (Total_Exposure_GC >= 0.8) & (Total_Exposure_GC <= 1)

# Calculating percentage for each Category
LL_GC = (np.sum(LL_GC_cond) / np.sum(~np.isnan(Total_Exposure_GC))) * 100
LH_GC = (np.sum(LH_GC_cond) / np.sum(~np.isnan(Total_Exposure_GC))) * 100
HL_GC = (np.sum(HL_GC_cond) / np.sum(~np.isnan(Total_Exposure_GC))) * 100
HH_GC = (np.sum(HH_GC_cond) / np.sum(~np.isnan(Total_Exposure_GC))) * 100
print("% LL_GC:", LL_GC)
print("% LH_GC:", LH_GC)
print("% HL_GC:", HL_GC)
print("% HH_GC:", HH_GC)


#BAL
# Applying the conditions to obtain values within the specified intervals
LL_BAL_cond = (Total_Exposure_BAL >= 0.2) & (Total_Exposure_BAL < 0.4)
LH_BAL_cond = (Total_Exposure_BAL >= 0.4) & (Total_Exposure_BAL < 0.6)
HL_BAL_cond = (Total_Exposure_BAL >= 0.6) & (Total_Exposure_BAL < 0.8)
HH_BAL_cond = (Total_Exposure_BAL >= 0.8) & (Total_Exposure_BAL <= 1)

# Calculating percentage for each Category
LL_BAL = (np.sum(LL_BAL_cond) / np.sum(~np.isnan(Total_Exposure_BAL))) * 100
LH_BAL = (np.sum(LH_BAL_cond) / np.sum(~np.isnan(Total_Exposure_BAL))) * 100
HL_BAL = (np.sum(HL_BAL_cond) / np.sum(~np.isnan(Total_Exposure_BAL))) * 100
HH_BAL = (np.sum(HH_BAL_cond) / np.sum(~np.isnan(Total_Exposure_BAL))) * 100
print("% LL_BAL:", LL_BAL)
print("% LH_BAL:", LH_BAL)
print("% HL_BAL:", HL_BAL)
print("% HH_BAL:", HH_BAL)


