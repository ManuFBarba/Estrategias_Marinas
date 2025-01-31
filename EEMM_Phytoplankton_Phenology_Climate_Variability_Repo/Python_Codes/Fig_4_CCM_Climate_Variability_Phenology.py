# -*- coding: utf-8 -*-
"""

###### Fig. 4. Causal Inference - Climate Extremes and Phyto Phenology ########

"""

#Loading required Python modules
import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

import pyEDM
import cartopy.crs as ccrs
import cartopy.feature as cft
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import cmocean as cm

from joblib import Parallel, delayed


## Loading CHL_Phenology_metrics.py


###############################################################################
## Causal Inference through EDM CCM (Sugihara et al., 2012)
###############################################################################

## Loading MHW Cum Intensity (from Fernández-Barba et al. (2024)), Windiness, and Max MLD datasets; and resampling to annual

MHWCumInt = np.load(r'E:\...\Annual_Data\MHWCumInt_Interp_Data/MHWCumInt_CAN.npy')
MHWCumInt = np.where(np.isnan(MHWCumInt), 0, MHWCumInt)

ds_MaxMLD = xr.open_dataset(r'E:\...\Monthly_Data\Monthly_MaxMLD/Monthly_MaxMLD_CAN.nc')
ds_MaxMLD = ds_MaxMLD.transpose('lat', 'lon', 'time')
MaxMLD = ds_MaxMLD.mlotst 
MaxMLD = MaxMLD.resample(time='YE').mean()
MaxMLD = np.asarray(MaxMLD)

ds_Windiness = xr.open_dataset(r'E:\...\Monthly_Data\Monthly_Windiness/Monthly_Windiness_CAN.nc')
ds_Windiness = ds_Windiness.transpose('latitude', 'longitude', 'time')
Windiness = ds_Windiness.wind_speed 
Windiness = Windiness.sortby('latitude', ascending=False)
Windiness = Windiness.resample(time='YE').sum()
Windiness = np.asarray(Windiness)
Windiness = np.where(Windiness == 0, np.nan, Windiness)


## Function to compute a pixelwise CCM test
def test_edm_for_point(column_series, target_series, embedding=6, lag=1):  # Adjust embedding and lags as needed
    if np.isnan(column_series).all() or np.isnan(target_series).all():
        return np.nan, np.nan

    data = {
        'Time': np.arange(len(column_series)),
        'Physical_Driver': column_series,
        'Phenology': target_series
    }
    df = pd.DataFrame(data).dropna()

    if len(df) < (embedding * lag):
        return np.nan, np.nan

    max_lib_size = len(df) - embedding * lag
    if max_lib_size < 10:
        return np.nan, np.nan

    libsizes = np.arange(10, max_lib_size-1, 5)  # Define a range of library sizes
    
    # Check if libsizes is empty
    if len(libsizes) == 0:
        return np.nan, np.nan
    
    result = pyEDM.CCM(
        dataFrame=df,
        E=embedding,
        tau=lag,
        columns="Physical_Driver",
        target="Phenology",
        libSizes=libsizes,
        sample=int(len(df))
    )
    
    if not result.empty:
        rho_column_to_target = result['Physical_Driver:Phenology'].max()
        rho_target_to_column = result['Phenology:Physical_Driver'].max()
        return rho_column_to_target, rho_target_to_column
    else:
        return np.nan, np.nan

## Function to process each pixel
def process_pixel(i, j):
    column_series = Windiness[i, j, :]
    target_series = SCR_CAN[i, j, :]
    return test_edm_for_point(column_series, target_series)

## Initialize output arrays
Windiness_to_SCR = np.full((734, 968), np.nan)
SCR_to_Windiness = np.full((734, 968), np.nan)

## Create a list of all pixel coordinates
pixel_indices = [(i, j) for i in range(734) for j in range(968)]

## Parallel processing using joblib
results = Parallel(n_jobs=-1)(delayed(process_pixel)(i, j) for i, j in pixel_indices)

## Assign results back to the output arrays
for idx, (rho_column_to_target, rho_target_to_column) in enumerate(results):
    i, j = pixel_indices[idx]
    Windiness_to_SCR[i, j] = rho_column_to_target
    SCR_to_Windiness[i, j] = rho_target_to_column
    print(f"Done: pixel (lon={i}, lat={j}), rho_column_to_target = {rho_column_to_target}, rho_target_to_column = {rho_target_to_column}")


## Save the results so far
np.save(r'E:\...\CCM_Outputs\SCR_xmap_Windiness/Windiness_to_SCR_CAN.npy', Windiness_to_SCR)
np.save(r'E:\...\CCM_Outputs\SCR_xmap_Windiness/SCR_to_Windiness_CAN.npy', SCR_to_Windiness)




## Fig. 4a (Bloom Max Chl-a xmap MHW Cum Intensity)
# Loading previously processed causality matrices
MHWCumInt_to_BMaxCHL_CAN = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MHWCumInt/MHWCumInt_to_BMaxCHL_CAN.npy')
MHWCumInt_to_BMaxCHL_NA = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MHWCumInt/MHWCumInt_to_BMaxCHL_NA.npy')
MHWCumInt_to_BMaxCHL_SA = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MHWCumInt/MHWCumInt_to_BMaxCHL_SA.npy')
MHWCumInt_to_BMaxCHL_AL = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MHWCumInt/MHWCumInt_to_BMaxCHL_AL.npy')
MHWCumInt_to_BMaxCHL_BAL = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MHWCumInt/MHWCumInt_to_BMaxCHL_BAL.npy')

# Representing CCM data
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection

land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.tempo
# cmap=cm.cm.matter
# cmap=cm.cm.algae
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MHWCumInt_to_BMaxCHL_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MHWCumInt_to_BMaxCHL_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MHWCumInt_to_BMaxCHL_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MHWCumInt_to_BMaxCHL_NA, levels=levels, cmap=cmap, transform=proj)


# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02)
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

cbar.set_label(r'Cross Map Skill [ρ]', fontsize=40)
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
axs.set_title('BMaxChl-a xmap MHW Cum Intensity', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MHWCumInt_to_BMaxCHL_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4a.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4b (Bloom Duration xmap MHW Cum Intensity)
# Loading previously processed causality matrices
MHWCumInt_to_BDur_CAN = np.load(r'E:\...\CCM_Outputs\BDur_xmap_MHWCumInt/MHWCumInt_to_BDur_CAN.npy')
MHWCumInt_to_BDur_NA = np.load(r'E:\...\CCM_Outputs\BDur_xmap_MHWCumInt/MHWCumInt_to_BDur_NA.npy')
MHWCumInt_to_BDur_SA = np.load(r'E:\...\CCM_Outputs\BDur_xmap_MHWCumInt/MHWCumInt_to_BDur_SA.npy')
MHWCumInt_to_BDur_AL = np.load(r'E:\...\CCM_Outputs\BDur_xmap_MHWCumInt/MHWCumInt_to_BDur_AL.npy')
MHWCumInt_to_BDur_BAL = np.load(r'E:\...\CCM_Outputs\BDur_xmap_MHWCumInt/MHWCumInt_to_BDur_BAL.npy')

# Representing CCM data
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.tempo
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MHWCumInt_to_BDur_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MHWCumInt_to_BDur_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MHWCumInt_to_BDur_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MHWCumInt_to_BDur_NA, levels=levels, cmap=cmap, transform=proj)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02)
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

cbar.set_label(r'Cross Map Skill [ρ]', fontsize=40)
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
axs.set_title('BDur xmap MHW Cum Intensity', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MHWCumInt_to_BDur_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4b.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4c (SCR xmap MHW Cum Intensity)
# Loading previously processed causality matrices
MHWCumInt_to_SCR_CAN = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MHWCumInt/MHWCumInt_to_SCR_CAN.npy')
MHWCumInt_to_SCR_NA = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MHWCumInt/MHWCumInt_to_SCR_NA.npy')
MHWCumInt_to_SCR_SA = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MHWCumInt/MHWCumInt_to_SCR_SA.npy')
MHWCumInt_to_SCR_AL = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MHWCumInt/MHWCumInt_to_SCR_AL.npy')
MHWCumInt_to_SCR_BAL = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MHWCumInt/MHWCumInt_to_SCR_BAL.npy')

# Representing CCM data
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.tempo
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MHWCumInt_to_SCR_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MHWCumInt_to_SCR_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MHWCumInt_to_SCR_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MHWCumInt_to_SCR_NA, levels=levels, cmap=cmap, transform=proj)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02)
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

cbar.set_label(r'Cross Map Skill [ρ]', fontsize=40)
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
axs.set_title('SCR xmap MHW Cum Intensity', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MHWCumInt_to_SCR_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4c.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4d (Bloom Max Chl-a xmap Windiness)
# Loading previously processed causality matrices
Windiness_to_BMaxCHL_CAN = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_Windiness/Windiness_to_BMaxCHL_CAN.npy')
Windiness_to_BMaxCHL_NA = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_Windiness/Windiness_to_BMaxCHL_NA.npy')
Windiness_to_BMaxCHL_SA = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_Windiness/Windiness_to_BMaxCHL_SA.npy')
Windiness_to_BMaxCHL_AL = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_Windiness/Windiness_to_BMaxCHL_AL.npy')
Windiness_to_BMaxCHL_BAL = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_Windiness/Windiness_to_BMaxCHL_BAL.npy')

# Representing CCM data
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.tempo
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, Windiness_to_BMaxCHL_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, Windiness_to_BMaxCHL_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, Windiness_to_BMaxCHL_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, Windiness_to_BMaxCHL_NA, levels=levels, cmap=cmap, transform=proj)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

el_2 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

el_3 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

el_4 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02)
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

cbar.set_label(r'Cross Map Skill [ρ]', fontsize=40)
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
axs.set_title('BMaxChl-a xmap Windiness', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, Windiness_to_BMaxCHL_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4d.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4e (Bloom Duration xmap Windiness) 
# Loading previously processed causality matrices
Windiness_to_BDur_CAN = np.load(r'E:\...\CCM_Outputs/BDur_xmap_Windiness/Windiness_to_BDur_CAN.npy')
Windiness_to_BDur_NA = np.load(r'E:\...\CCM_Outputs/BDur_xmap_Windiness/Windiness_to_BDur_NA.npy')
Windiness_to_BDur_SA = np.load(r'E:\...\CCM_Outputs/BDur_xmap_Windiness/Windiness_to_BDur_SA.npy')
Windiness_to_BDur_AL = np.load(r'E:\...\CCM_Outputs/BDur_xmap_Windiness/Windiness_to_BDur_AL.npy')
Windiness_to_BDur_BAL = np.load(r'E:\...\CCM_Outputs/BDur_xmap_Windiness/Windiness_to_BDur_BAL.npy')

# Representing CCM data
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.tempo
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, Windiness_to_BDur_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, Windiness_to_BDur_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, Windiness_to_BDur_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, Windiness_to_BDur_NA, levels=levels, cmap=cmap, transform=proj)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02)
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

cbar.set_label(r'Cross Map Skill [ρ]', fontsize=40)
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
axs.set_title('BDur xmap Windiness', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, Windiness_to_BDur_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4e.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4f (SCR xmap Windiness) 
# Loading previously processed causality matrices
Windiness_to_SCR_CAN = np.load(r'E:\...\CCM_Outputs/SCR_xmap_Windiness/Windiness_to_SCR_CAN.npy')
Windiness_to_SCR_NA = np.load(r'E:\...\CCM_Outputs/SCR_xmap_Windiness/Windiness_to_SCR_NA.npy')
Windiness_to_SCR_SA = np.load(r'E:\...\CCM_Outputs/SCR_xmap_Windiness/Windiness_to_SCR_SA.npy')
Windiness_to_SCR_AL = np.load(r'E:\...\CCM_Outputs/SCR_xmap_Windiness/Windiness_to_SCR_AL.npy')
Windiness_to_SCR_BAL = np.load(r'E:\...\CCM_Outputs/SCR_xmap_Windiness/Windiness_to_SCR_BAL.npy')

# Representing CCM data
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.tempo
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, Windiness_to_SCR_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, Windiness_to_SCR_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, Windiness_to_SCR_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, Windiness_to_SCR_NA, levels=levels, cmap=cmap, transform=proj)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02)
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

cbar.set_label(r'Cross Map Skill [ρ]', fontsize=40)
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
axs.set_title('SCR xmap Windiness', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, Windiness_to_SCR_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4f.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4g (Bloom Max Chl-a xmap Max MLD) 
# Loading previously processed causality matrices
MaxMLD_to_BMaxCHL_CAN = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MaxMLD/MaxMLD_to_BMaxCHL_CAN.npy')
MaxMLD_to_BMaxCHL_NA = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MaxMLD/MaxMLD_to_BMaxCHL_NA.npy')
MaxMLD_to_BMaxCHL_SA = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MaxMLD/MaxMLD_to_BMaxCHL_SA.npy')
MaxMLD_to_BMaxCHL_AL = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MaxMLD/MaxMLD_to_BMaxCHL_AL.npy')
MaxMLD_to_BMaxCHL_BAL = np.load(r'E:\...\CCM_Outputs\BMaxCHL_xmap_MaxMLD/MaxMLD_to_BMaxCHL_BAL.npy')

# Representing CCM data
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.tempo
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MaxMLD_to_BMaxCHL_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MaxMLD_to_BMaxCHL_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MaxMLD_to_BMaxCHL_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MaxMLD_to_BMaxCHL_NA, levels=levels, cmap=cmap, transform=proj)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02)
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

cbar.set_label(r'Cross Map Skill [ρ]', fontsize=40)
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
axs.set_title('BMaxChl-a xmap Max MLD', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MaxMLD_to_BMaxCHL_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4g.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4h (Bloom Duration xmap Max MLD) 
# Loading previously processed causality matrices
MaxMLD_to_BDur_CAN = np.load(r'E:\...\CCM_Outputs/BDur_xmap_MaxMLD/MaxMLD_to_BDur_CAN.npy')
MaxMLD_to_BDur_NA = np.load(r'E:\...\CCM_Outputs/BDur_xmap_MaxMLD/MaxMLD_to_BDur_NA.npy')
MaxMLD_to_BDur_SA = np.load(r'E:\...\CCM_Outputs/BDur_xmap_MaxMLD/MaxMLD_to_BDur_SA.npy')
MaxMLD_to_BDur_AL = np.load(r'E:\...\CCM_Outputs/BDur_xmap_MaxMLD/MaxMLD_to_BDur_AL.npy')
MaxMLD_to_BDur_BAL = np.load(r'E:\...\CCM_Outputs/BDur_xmap_MaxMLD/MaxMLD_to_BDur_BAL.npy')

# Representing CCM data
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.tempo
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MaxMLD_to_BDur_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MaxMLD_to_BDur_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MaxMLD_to_BDur_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MaxMLD_to_BDur_NA, levels=levels, cmap=cmap, transform=proj)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02)
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

cbar.set_label(r'Cross Map Skill [ρ]', fontsize=40)
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
axs.set_title('BDur xmap Max MLD', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MaxMLD_to_BDur_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4h.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4i (SCR xmap Max MLD) 
# Loading previously processed causality matrices
MaxMLD_to_SCR_CAN = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MaxMLD/MaxMLD_to_SCR_CAN.npy')
MaxMLD_to_SCR_NA = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MaxMLD/MaxMLD_to_SCR_NA.npy')
MaxMLD_to_SCR_SA = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MaxMLD/MaxMLD_to_SCR_SA.npy')
MaxMLD_to_SCR_AL = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MaxMLD/MaxMLD_to_SCR_AL.npy')
MaxMLD_to_SCR_BAL = np.load(r'E:\...\CCM_Outputs/SCR_xmap_MaxMLD/MaxMLD_to_SCR_BAL.npy')

# Representing CCM data
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.tempo
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MaxMLD_to_SCR_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MaxMLD_to_SCR_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MaxMLD_to_SCR_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MaxMLD_to_SCR_NA, levels=levels, cmap=cmap, transform=proj)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02)
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

cbar.set_label(r'Cross Map Skill [ρ]', fontsize=40)
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
axs.set_title('SCR xmap Max MLD', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MaxMLD_to_SCR_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4i.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Loading Data for 2D Probability Histograms
BMaxCHL_SA = np.load(r'E:\...\Phenology_Metrics\SA/bloom_max_SA.npy')
BMaxCHL_NA = np.load(r'E:\...\Phenology_Metrics\NA/bloom_max_NA.npy')
BMaxCHL_AL = np.load(r'E:\...\Phenology_Metrics\AL/bloom_max_AL.npy')
BMaxCHL_BAL = np.load(r'E:\...\Phenology_Metrics\BAL/bloom_max_BAL.npy')
BMaxCHL_CAN = np.load(r'E:\...\Phenology_Metrics\CAN/bloom_max_CAN.npy')
BDur_SA = np.load(r'E:\...\Phenology_Metrics\SA/bloom_dur_SA.npy')
BDur_NA = np.load(r'E:\...\Phenology_Metrics\NA/bloom_dur_NA.npy')
BDur_AL = np.load(r'E:\...\Phenology_Metrics\AL/bloom_dur_AL.npy')
BDur_BAL = np.load(r'E:\...\Phenology_Metrics\BAL/bloom_dur_BAL.npy')
BDur_CAN = np.load(r'E:\...\Phenology_Metrics\CAN/bloom_dur_CAN.npy')
SCR_SA = np.load(r'E:\...\Phenology_Metrics\SA/SCR_SA.npy')
SCR_NA = np.load(r'E:\...\Phenology_Metrics\NA/SCR_NA.npy')
SCR_AL = np.load(r'E:\...\Phenology_Metrics\AL/SCR_AL.npy')
SCR_BAL = np.load(r'E:\...\Phenology_Metrics\BAL/SCR_BAL.npy')
SCR_CAN = np.load(r'E:\...\Phenology_Metrics\CAN/SCR_CAN.npy')

MHWCumInt_SA = np.load(r'E:\...\Annual_Data\MHWCumInt_Interp_Data/MHWCumInt_SA.npy')
MHWCumInt_NA = np.load(r'E:\...\Annual_Data\MHWCumInt_Interp_Data/MHWCumInt_NA.npy')
MHWCumInt_AL = np.load(r'E:\...\Annual_Data\MHWCumInt_Interp_Data/MHWCumInt_AL.npy')
MHWCumInt_BAL = np.load(r'E:\...\Annual_Data\MHWCumInt_Interp_Data/MHWCumInt_BAL.npy')
MHWCumInt_CAN = np.load(r'E:\...\Annual_Data\MHWCumInt_Interp_Data/MHWCumInt_CAN.npy')

Windiness_SA = np.load(r'E:\...\Annual_Data\Windiness/Windiness_SA.npy')
Windiness_NA = np.load(r'E:\...\Annual_Data\Windiness/Windiness_NA.npy')
Windiness_AL = np.load(r'E:\...\Annual_Data\Windiness/Windiness_AL.npy')
Windiness_BAL = np.load(r'E:\...\Annual_Data\Windiness/Windiness_BAL.npy')
Windiness_CAN = np.load(r'E:\...\Annual_Data\Windiness/Windiness_CAN.npy')

MaxMLD_SA = np.load(r'E:\...\Annual_Data\MaxMLD/MaxMLD_SA.npy')
MaxMLD_NA = np.load(r'E:\...\Annual_Data\MaxMLD/MaxMLD_NA.npy')
MaxMLD_AL = np.load(r'E:\...\Annual_Data\MaxMLD/MaxMLD_AL.npy')
MaxMLD_BAL = np.load(r'E:\...\Annual_Data\MaxMLD/MaxMLD_BAL.npy')
MaxMLD_CAN = np.load(r'E:\...\Annual_Data\MaxMLD/MaxMLD_CAN.npy')


BMaxCHL_SA = BMaxCHL_SA[..., :-1]
BMaxCHL_NA = BMaxCHL_NA[..., :-1]
BMaxCHL_AL = BMaxCHL_AL[..., :-1]
BMaxCHL_BAL = BMaxCHL_BAL[..., :-1]
BMaxCHL_CAN = BMaxCHL_CAN[..., :-1]
BDur_SA = BDur_SA[..., :-1]
BDur_NA = BDur_NA[..., :-1]
BDur_AL = BDur_AL[..., :-1]
BDur_BAL = BDur_BAL[..., :-1]
BDur_CAN = BDur_CAN[..., :-1]
SCR_SA = SCR_SA[..., :-1]
SCR_NA = SCR_NA[..., :-1]
SCR_AL = SCR_AL[..., :-1]
SCR_BAL = SCR_BAL[..., :-1]
SCR_CAN = SCR_CAN[..., :-1]

Windiness_SA = Windiness_SA[..., :-1]
Windiness_NA = Windiness_NA[..., :-1]
Windiness_AL = Windiness_AL[..., :-1]
Windiness_BAL = Windiness_BAL[..., :-1]
Windiness_CAN = Windiness_CAN[..., :-1]
MaxMLD_SA = MaxMLD_SA[..., :-1]
MaxMLD_NA = MaxMLD_NA[..., :-1]
MaxMLD_AL = MaxMLD_AL[..., :-1]
MaxMLD_BAL = MaxMLD_BAL[..., :-1]
MaxMLD_CAN = MaxMLD_CAN[..., :-1]



## Fig. 4j (Climate Variability - Bloom Max Chl-a 2D Probability Histogram)
bloom_data = [BMaxCHL_SA, BMaxCHL_NA, BMaxCHL_AL, BMaxCHL_BAL, BMaxCHL_CAN]
mhw_data = [MHWCumInt_SA, MHWCumInt_NA, MHWCumInt_AL, MHWCumInt_BAL, MHWCumInt_CAN]
windiness_data = [Windiness_SA, Windiness_NA, Windiness_AL, Windiness_BAL, Windiness_CAN]
mld_data = [MaxMLD_SA, MaxMLD_NA, MaxMLD_AL, MaxMLD_BAL, MaxMLD_CAN]

regions = ['NA', 'AL', 'CAN', 'SA', 'BAL']

variables = ['MHW Cum Intensity\n[ºC·days]', 'Windiness\n[days]', 'Max MLD\n[m]']
data_pairs = [mhw_data, windiness_data, mld_data]

fig, axs = plt.subplots(len(regions), len(variables), figsize=(20, 20), sharey='row')
fig.set_constrained_layout_pads(w_pad=3.0, h_pad=3.0, hspace=0.2, wspace=0.2)

for ax_row in axs:
    for ax in ax_row:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(40)

norm = plt.Normalize(vmin=0, vmax=0.02)
sm = plt.cm.ScalarMappable(cmap=cm.cm.rain, norm=norm)
sm.set_array([])

# [(x_min, x_max, y_min, y_max), ...]
axis_limits = [
    (5, 70, 0.2, 5),   # Subplot (0, 0) NA
    (65, 130, 0.2, 5),      # Subplot (0, 1) NA
    (10, 80, 0.2, 5),    # Subplot (0, 2) NA
    (5, 60, 0.2, 3),     # Subplot (1, 0) AL
    (70, 130, 0.2, 3),     # Subplot (1, 1) AL
    (45, 130, 0.2, 3),     # Subplot (1, 2) AL
    (5, 75, 0.5, 4),     # Subplot (2, 0) CAN
    (70, 125, 0.5, 4),     # Subplot (2, 1) CAN
    (15, 55, 0.5, 4),    # Subplot (2, 2) CAN
    (5, 100, 0.3, 2),    # Subplot (3, 0) SA
    (70, 125, 0.3, 2),    # Subplot (3, 1) SA
    (20, 80, 0.3, 2),    # Subplot (3, 2) SA
    (5, 95, 0.1, 1),    # Subplot (4, 0) BAL
    (55, 140, 0.1, 1),    # Subplot (4, 1) BAL
    (40, 100, 0.1, 1),    # Subplot (4, 2) BAL
]

for i, (bloom, region) in enumerate(zip(bloom_data, regions)):
    for j, (var_data, var_name) in enumerate(zip(data_pairs, variables)):
        bloom_flat = bloom.flatten()
        var_flat = var_data[i].flatten()
        valid_indices = np.isfinite(bloom_flat) & np.isfinite(var_flat)
        bloom_flat = bloom_flat[valid_indices]
        var_flat = var_flat[valid_indices]

        h, xedges, yedges, img = axs[i, j].hist2d(
            var_flat, bloom_flat, bins=100, cmap=cm.cm.rain, density=True, norm=norm)

        x_min, x_max, y_min, y_max = axis_limits[i * len(variables) + j]
        axs[i, j].set_xlim([x_min, x_max])
        axs[i, j].set_ylim([y_min, y_max])

        axs[i, j].xaxis.set_major_locator(MaxNLocator(4))  
        axs[i, j].yaxis.set_major_locator(MaxNLocator(3))  
        axs[i, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        
        if i == len(regions) - 1:
            axs[i, j].set_xlabel(var_name, fontsize=40)
            
        if j == 0:
            axs[i, j].set_ylabel(
                f'$\\bf{{{region}}}$\nBMaxChl-a\n$[mg \cdot m^{{-3}}]$', 
                fontsize=40)  

plt.tight_layout()
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4j.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4k (Climate Variability - Bloom Duration 2D Probability Histogram)
bloom_data = [BDur_SA, BDur_NA, BDur_AL, BDur_BAL, BDur_CAN]
mhw_data = [MHWCumInt_SA, MHWCumInt_NA, MHWCumInt_AL, MHWCumInt_BAL, MHWCumInt_CAN]
windiness_data = [Windiness_SA, Windiness_NA, Windiness_AL, Windiness_BAL, Windiness_CAN]
mld_data = [MaxMLD_SA, MaxMLD_NA, MaxMLD_AL, MaxMLD_BAL, MaxMLD_CAN]

regions = ['NA', 'AL', 'CAN', 'SA', 'BAL']

variables = ['MHW Cum Intensity\n[ºC·days]', 'Windiness\n[days]', 'Max MLD\n[m]']
data_pairs = [mhw_data, windiness_data, mld_data]

fig, axs = plt.subplots(len(regions), len(variables), figsize=(20, 20), sharey='row')
fig.set_constrained_layout_pads(w_pad=3.0, h_pad=3.0, hspace=0.2, wspace=0.2)

for ax_row in axs:
    for ax in ax_row:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(40)

norm = plt.Normalize(vmin=0, vmax=0.0004)
sm = plt.cm.ScalarMappable(cmap=cm.cm.rain, norm=norm)
sm.set_array([])

# [(x_min, x_max, y_min, y_max), ...]
axis_limits = [
    (5, 70, 10, 165),   # Subplot (0, 0) NA
    (65, 130, 10, 165),      # Subplot (0, 1) NA
    (10, 80, 10, 165),    # Subplot (0, 2) NA
    (5, 60, 25, 170),     # Subplot (1, 0) AL
    (70, 130, 25, 170),     # Subplot (1, 1) AL
    (45, 130, 25, 170),     # Subplot (1, 2) AL
    (5, 75, 20, 160),     # Subplot (2, 0) CAN
    (70, 125, 20, 160),     # Subplot (2, 1) CAN
    (15, 55, 20, 160),    # Subplot (2, 2) CAN
    (5, 100, 20, 155),    # Subplot (3, 0) SA
    (70, 125, 20, 155),    # Subplot (3, 1) SA
    (20, 75, 20, 155),    # Subplot (3, 2) SA
    (5, 60, 15, 150),    # Subplot (4, 0) BAL
    (60, 130, 15, 150),    # Subplot (4, 1) BAL
    (40, 90, 15, 150),    # Subplot (4, 2) BAL
]

for i, (bloom, region) in enumerate(zip(bloom_data, regions)):
    for j, (var_data, var_name) in enumerate(zip(data_pairs, variables)):
        bloom_flat = bloom.flatten()
        var_flat = var_data[i].flatten()
        valid_indices = np.isfinite(bloom_flat) & np.isfinite(var_flat)
        bloom_flat = bloom_flat[valid_indices]
        var_flat = var_flat[valid_indices]

        h, xedges, yedges, img = axs[i, j].hist2d(
            var_flat, bloom_flat, bins=100, cmap=cm.cm.rain, density=True, norm=norm)

        x_min, x_max, y_min, y_max = axis_limits[i * len(variables) + j]
        axs[i, j].set_xlim([x_min, x_max])
        axs[i, j].set_ylim([y_min, y_max])

        axs[i, j].xaxis.set_major_locator(MaxNLocator(4))  
        axs[i, j].yaxis.set_major_locator(MaxNLocator(3))  
        axs[i, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
        
        if i == len(regions) - 1:
            axs[i, j].set_xlabel(var_name, fontsize=40)
        
        if j == 0:
            axs[i, j].set_ylabel(
                f'BDur\n$[days]$', 
                fontsize=40)

plt.tight_layout()
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4k.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 4l (Climate Variability - SCR 2D Probability Histogram)
bloom_data = [SCR_SA, SCR_NA, SCR_AL, SCR_BAL, SCR_CAN]
mhw_data = [MHWCumInt_SA, MHWCumInt_NA, MHWCumInt_AL, MHWCumInt_BAL, MHWCumInt_CAN]
windiness_data = [Windiness_SA, Windiness_NA, Windiness_AL, Windiness_BAL, Windiness_CAN]
mld_data = [MaxMLD_SA, MaxMLD_NA, MaxMLD_AL, MaxMLD_BAL, MaxMLD_CAN]

regions = ['NA', 'AL', 'CAN', 'SA', 'BAL']

variables = ['MHW Cum Intensity\n[ºC·days]', 'Windiness\n[days]', 'Max MLD\n[m]']
data_pairs = [mhw_data, windiness_data, mld_data]

fig, axs = plt.subplots(len(regions), len(variables), figsize=(20, 20), sharey='row')
fig.set_constrained_layout_pads(w_pad=3.0, h_pad=3.0, hspace=0.2, wspace=0.2)

for ax_row in axs:
    for ax in ax_row:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(40)

norm = plt.Normalize(vmin=0, vmax=0.06)
sm = plt.cm.ScalarMappable(cmap=cm.cm.rain, norm=norm)
sm.set_array([])

#[(x_min, x_max, y_min, y_max), ...]
axis_limits = [
    (5, 70, 0.05, 0.8),   # Subplot (0, 0) NA
    (65, 130, 0.05, 0.8),      # Subplot (0, 1) NA
    (10, 80, 0.05, 0.8),    # Subplot (0, 2) NA
    (5, 60, 0.15, 0.9),     # Subplot (1, 0) AL
    (70, 130, 0.15, 0.9),     # Subplot (1, 1) AL
    (45, 130, 0.15, 0.9),     # Subplot (1, 2) AL
    (5, 75, 0.05, 0.8),     # Subplot (2, 0) CAN
    (70, 125, 0.05, 0.8),     # Subplot (2, 1) CAN
    (15, 55, 0.05, 0.8),    # Subplot (2, 2) CAN
    (5, 100, 0.2, 0.9),    # Subplot (3, 0) SA
    (70, 125, 0.2, 0.9),    # Subplot (3, 1) SA
    (20, 75, 0.2, 0.9),    # Subplot (3, 2) SA
    (5, 55, 0.2, 0.95),    # Subplot (4, 0) BAL
    (65, 130, 0.2, 0.95),    # Subplot (4, 1) BAL
    (45, 90, 0.2, 0.95),    # Subplot (4, 2) BAL
]

for i, (bloom, region) in enumerate(zip(bloom_data, regions)):
    for j, (var_data, var_name) in enumerate(zip(data_pairs, variables)):
        bloom_flat = bloom.flatten()
        var_flat = var_data[i].flatten()
        valid_indices = np.isfinite(bloom_flat) & np.isfinite(var_flat)
        bloom_flat = bloom_flat[valid_indices]
        var_flat = var_flat[valid_indices]

        h, xedges, yedges, img = axs[i, j].hist2d(
            var_flat, bloom_flat, bins=100, cmap=cm.cm.rain, density=True, norm=norm)

        x_min, x_max, y_min, y_max = axis_limits[i * len(variables) + j]
        axs[i, j].set_xlim([x_min, x_max])
        axs[i, j].set_ylim([y_min, y_max])

        # Ajustar la cantidad de ticks en cada eje
        axs[i, j].xaxis.set_major_locator(MaxNLocator(4))  
        axs[i, j].yaxis.set_major_locator(MaxNLocator(3))  
        axs[i, j].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        
        if i == len(regions) - 1:
            axs[i, j].set_xlabel(var_name, fontsize=40)
        
        if j == 0:
            axs[i, j].set_ylabel(
                f'SCR\n$[r]$', 
                fontsize=40)

# cbar_ax = fig.add_axes([1, 0.15, 0.025, 0.75])  # [left, bottom, width, height]
# cbar = plt.colorbar(sm, cax=cbar_ax, format=ticker.FormatStrFormatter('%.2f'), extend='max')
# cbar.ax.tick_params(axis='y', size=16, direction='in', labelsize=45)
# cbar.ax.minorticks_off()
# cbar.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
# cbar.set_label(r'Probability Density', fontsize=45)

plt.tight_layout()
plt.show()

outfile = r'E:\...\Figures\Fig_4\Fig_4l.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')

