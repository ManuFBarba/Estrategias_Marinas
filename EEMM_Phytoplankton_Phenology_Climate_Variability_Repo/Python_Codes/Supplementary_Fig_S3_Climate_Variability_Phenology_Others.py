# -*- coding: utf-8 -*-
"""

###### Supplementary Fig. S3. Causal Inference - Climate Extremes and Phyto Phenology (Others) ########

"""

#Loading required Python modules
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

## Loading Annual MHW Cum Intensity (from Fernández-Barba et al., 2024), Windiness, and Max MLD datasets
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

BInit_SA = bloom_ini_SA#[..., :-1] #To align with MWH Cum Int (from 1998 to 2022)
BInit_NA = bloom_ini_NA#[..., :-1]
BInit_AL = bloom_ini_AL#[..., :-1]
BInit_BAL = bloom_ini_BAL#[..., :-1]
BInit_CAN = bloom_ini_CAN#[..., :-1]


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

    libsizes = np.arange(10, max_lib_size-1, 2)
    
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
    column_series = MHWCumInt_SA[i, j, :]
    target_series = BInit_SA[i, j, :]
    return test_edm_for_point(column_series, target_series)

## Initialize output arrays
MHWCumInt_to_BInit = np.full((138, 154), np.nan)
BInit_to_MHWCumInt = np.full((138, 154), np.nan)

## Create a list of all pixel coordinates
pixel_indices = [(i, j) for i in range(138) for j in range(154)]

## Parallel processing using joblib
results = Parallel(n_jobs=-1)(delayed(process_pixel)(i, j) for i, j in pixel_indices)

## Assign results back to the output arrays
for idx, (rho_column_to_target, rho_target_to_column) in enumerate(results):
    i, j = pixel_indices[idx]
    MHWCumInt_to_BInit[i, j] = rho_column_to_target
    BInit_to_MHWCumInt[i, j] = rho_target_to_column
    print(f"Done: pixel (lon={i}, lat={j}), rho_column_to_target = {rho_column_to_target}, rho_target_to_column = {rho_target_to_column}")

## Save the results so far
np.save(r'E:\...\CCM_Outputs\BInit_xmap_MHWCumInt/MHWCumInt_to_BInit_SA.npy', MHWCumInt_to_BInit)
np.save(r'E:\...\CCM_Outputs\BInit_xmap_MHWCumInt/BInit_to_MHWCumInt_SA.npy', BInit_to_MHWCumInt)



## Supplementary Fig. S3a (Bloom Initiation xmap MHW Cum Intensity)
# Loading previously processed causality matrices
MHWCumInt_to_BInit_CAN = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MHWCumInt/MHWCumInt_to_BInit_CAN.npy')
MHWCumInt_to_BInit_NA = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MHWCumInt/MHWCumInt_to_BInit_NA.npy')
MHWCumInt_to_BInit_SA = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MHWCumInt/MHWCumInt_to_BInit_SA.npy')
MHWCumInt_to_BInit_AL = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MHWCumInt/MHWCumInt_to_BInit_AL.npy')
MHWCumInt_to_BInit_BAL = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MHWCumInt/MHWCumInt_to_BInit_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MHWCumInt_to_BInit_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MHWCumInt_to_BInit_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MHWCumInt_to_BInit_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MHWCumInt_to_BInit_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BInit xmap MHW Cum Intensity', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MHWCumInt_to_BInit_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3a.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3b (Bloom Initiation xmap Windiness)
# Loading previously processed causality matrices
Windiness_to_BInit_CAN = np.load(r'E:\...\CCM_Outputs\BInit_xmap_Windiness/Windiness_to_BInit_CAN.npy')
Windiness_to_BInit_NA = np.load(r'E:\...\CCM_Outputs\BInit_xmap_Windiness/Windiness_to_BInit_NA.npy')
Windiness_to_BInit_SA = np.load(r'E:\...\CCM_Outputs\BInit_xmap_Windiness/Windiness_to_BInit_SA.npy')
Windiness_to_BInit_AL = np.load(r'E:\...\CCM_Outputs\BInit_xmap_Windiness/Windiness_to_BInit_AL.npy')
Windiness_to_BInit_BAL = np.load(r'E:\...\CCM_Outputs\BInit_xmap_Windiness/Windiness_to_BInit_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, Windiness_to_BInit_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, Windiness_to_BInit_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, Windiness_to_BInit_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, Windiness_to_BInit_NA, levels=levels, cmap=cmap, transform=proj)

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
axs.set_title('BInit xmap Windiness', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, Windiness_to_BInit_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3b.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3c (Bloom Initiation xmap Max MLD)
# Loading previously processed causality matrices
MaxMLD_to_BInit_CAN = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MaxMLD/MaxMLD_to_BInit_CAN.npy')
MaxMLD_to_BInit_NA = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MaxMLD/MaxMLD_to_BInit_NA.npy')
MaxMLD_to_BInit_SA = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MaxMLD/MaxMLD_to_BInit_SA.npy')
MaxMLD_to_BInit_AL = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MaxMLD/MaxMLD_to_BInit_AL.npy')
MaxMLD_to_BInit_BAL = np.load(r'E:\...\CCM_Outputs\BInit_xmap_MaxMLD/MaxMLD_to_BInit_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MaxMLD_to_BInit_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MaxMLD_to_BInit_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MaxMLD_to_BInit_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MaxMLD_to_BInit_NA, levels=levels, cmap=cmap, transform=proj)

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
axs.set_title('BInit xmap Max MLD', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MaxMLD_to_BInit_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3c.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3d (Bloom Termination xmap MHW Cum Intensity)
# Loading previously processed causality matrices
MHWCumInt_to_BTerm_CAN = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MHWCumInt/MHWCumInt_to_BTerm_CAN.npy')
MHWCumInt_to_BTerm_NA = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MHWCumInt/MHWCumInt_to_BTerm_NA.npy')
MHWCumInt_to_BTerm_SA = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MHWCumInt/MHWCumInt_to_BTerm_SA.npy')
MHWCumInt_to_BTerm_AL = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MHWCumInt/MHWCumInt_to_BTerm_AL.npy')
MHWCumInt_to_BTerm_BAL = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MHWCumInt/MHWCumInt_to_BTerm_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MHWCumInt_to_BTerm_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MHWCumInt_to_BTerm_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MHWCumInt_to_BTerm_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MHWCumInt_to_BTerm_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BTerm xmap MHW Cum Intensity', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MHWCumInt_to_BTerm_CAN, levels=levels, cmap=cmap, transform=proj)
# cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, BTerm_to_MHWCumInt_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3d.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3e (Bloom Termination xmap Windiness)
# Loading previously processed causality matrices
Windiness_to_BTerm_CAN = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_Windiness/Windiness_to_BTerm_CAN.npy')
Windiness_to_BTerm_NA = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_Windiness/Windiness_to_BTerm_NA.npy')
Windiness_to_BTerm_SA = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_Windiness/Windiness_to_BTerm_SA.npy')
Windiness_to_BTerm_AL = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_Windiness/Windiness_to_BTerm_AL.npy')
Windiness_to_BTerm_BAL = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_Windiness/Windiness_to_BTerm_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, Windiness_to_BTerm_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, Windiness_to_BTerm_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, Windiness_to_BTerm_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, Windiness_to_BTerm_NA, levels=levels, cmap=cmap, transform=proj)

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
axs.set_title('BTerm xmap Windiness', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, Windiness_to_BTerm_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3e.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')




## Supplementary Fig. S3f (Bloom Termination xmap Max MLD)
# Loading previously processed causality matrices
MaxMLD_to_BTerm_CAN = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MaxMLD/MaxMLD_to_BTerm_CAN.npy')
MaxMLD_to_BTerm_NA = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MaxMLD/MaxMLD_to_BTerm_NA.npy')
MaxMLD_to_BTerm_SA = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MaxMLD/MaxMLD_to_BTerm_SA.npy')
MaxMLD_to_BTerm_AL = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MaxMLD/MaxMLD_to_BTerm_AL.npy')
MaxMLD_to_BTerm_BAL = np.load(r'E:\...\CCM_Outputs\BTerm_xmap_MaxMLD/MaxMLD_to_BTerm_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MaxMLD_to_BTerm_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MaxMLD_to_BTerm_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MaxMLD_to_BTerm_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MaxMLD_to_BTerm_NA, levels=levels, cmap=cmap, transform=proj)

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
axs.set_title('BTerm xmap Max MLD', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MaxMLD_to_BTerm_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3f.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3g (Bloom Cumulative Chl-a xmap MHW Cum Intensity)
# Loading previously processed causality matrices
MHWCumInt_to_BCumChla_CAN = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MHWCumInt/MHWCumInt_to_BCumChla_CAN.npy')
MHWCumInt_to_BCumChla_NA = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MHWCumInt/MHWCumInt_to_BCumChla_NA.npy')
MHWCumInt_to_BCumChla_SA = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MHWCumInt/MHWCumInt_to_BCumChla_SA.npy')
MHWCumInt_to_BCumChla_AL = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MHWCumInt/MHWCumInt_to_BCumChla_AL.npy')
MHWCumInt_to_BCumChla_BAL = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MHWCumInt/MHWCumInt_to_BCumChla_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MHWCumInt_to_BCumChla_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MHWCumInt_to_BCumChla_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MHWCumInt_to_BCumChla_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MHWCumInt_to_BCumChla_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BCumChl-a xmap MHW Cum Intensity', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MHWCumInt_to_BCumChla_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3g.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3h (Bloom Cumulative Chl-a xmap Windiness)
# Loading previously processed causality matrices
Windiness_to_BCumChla_CAN = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_Windiness/Windiness_to_BCumChla_CAN.npy')
Windiness_to_BCumChla_NA = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_Windiness/Windiness_to_BCumChla_NA.npy')
Windiness_to_BCumChla_SA = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_Windiness/Windiness_to_BCumChla_SA.npy')
Windiness_to_BCumChla_AL = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_Windiness/Windiness_to_BCumChla_AL.npy')
Windiness_to_BCumChla_BAL = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_Windiness/Windiness_to_BCumChla_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, Windiness_to_BCumChla_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, Windiness_to_BCumChla_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, Windiness_to_BCumChla_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, Windiness_to_BCumChla_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BCumChl-a xmap Windiness', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, Windiness_to_BCumChla_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3h.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3i (Bloom Cumulative Chl-a xmap Max MLD)
# Loading previously processed causality matrices
MaxMLD_to_BCumChla_CAN = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MaxMLD/MaxMLD_to_BCumChla_CAN.npy')
MaxMLD_to_BCumChla_NA = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MaxMLD/MaxMLD_to_BCumChla_NA.npy')
MaxMLD_to_BCumChla_SA = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MaxMLD/MaxMLD_to_BCumChla_SA.npy')
MaxMLD_to_BCumChla_AL = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MaxMLD/MaxMLD_to_BCumChla_AL.npy')
MaxMLD_to_BCumChla_BAL = np.load(r'E:\...\CCM_Outputs\BCumChla_xmap_MaxMLD/MaxMLD_to_BCumChla_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MaxMLD_to_BCumChla_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MaxMLD_to_BCumChla_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MaxMLD_to_BCumChla_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MaxMLD_to_BCumChla_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BCumChl-a xmap Max MLD', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MaxMLD_to_BCumChla_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3i.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3j (Bloom Peak xmap MHW Cum Intensity)
# Loading previously processed causality matrices
MHWCumInt_to_BPeak_CAN = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MHWCumInt/MHWCumInt_to_BPeak_CAN.npy')
MHWCumInt_to_BPeak_NA = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MHWCumInt/MHWCumInt_to_BPeak_NA.npy')
MHWCumInt_to_BPeak_SA = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MHWCumInt/MHWCumInt_to_BPeak_SA.npy')
MHWCumInt_to_BPeak_AL = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MHWCumInt/MHWCumInt_to_BPeak_AL.npy')
MHWCumInt_to_BPeak_BAL = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MHWCumInt/MHWCumInt_to_BPeak_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MHWCumInt_to_BPeak_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MHWCumInt_to_BPeak_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MHWCumInt_to_BPeak_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MHWCumInt_to_BPeak_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BPeak xmap MHW Cum Intensity', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MHWCumInt_to_BPeak_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3j.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3k (Bloom Peak xmap Windiness)
# Loading previously processed causality matrices
Windiness_to_BPeak_CAN = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_Windiness/Windiness_to_BPeak_CAN.npy')
Windiness_to_BPeak_NA = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_Windiness/Windiness_to_BPeak_NA.npy')
Windiness_to_BPeak_SA = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_Windiness/Windiness_to_BPeak_SA.npy')
Windiness_to_BPeak_AL = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_Windiness/Windiness_to_BPeak_AL.npy')
Windiness_to_BPeak_BAL = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_Windiness/Windiness_to_BPeak_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, Windiness_to_BPeak_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, Windiness_to_BPeak_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, Windiness_to_BPeak_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, Windiness_to_BPeak_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BPeak xmap Windiness', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, Windiness_to_BPeak_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3k.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3l (Bloom Peak xmap Max MLD)
# Loading previously processed causality matrices
MaxMLD_to_BPeak_CAN = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MaxMLD/MaxMLD_to_BPeak_CAN.npy')
MaxMLD_to_BPeak_NA = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MaxMLD/MaxMLD_to_BPeak_NA.npy')
MaxMLD_to_BPeak_SA = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MaxMLD/MaxMLD_to_BPeak_SA.npy')
MaxMLD_to_BPeak_AL = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MaxMLD/MaxMLD_to_BPeak_AL.npy')
MaxMLD_to_BPeak_BAL = np.load(r'E:\...\CCM_Outputs\BPeak_xmap_MaxMLD/MaxMLD_to_BPeak_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MaxMLD_to_BPeak_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MaxMLD_to_BPeak_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MaxMLD_to_BPeak_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MaxMLD_to_BPeak_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BPeak xmap Max MLD', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MaxMLD_to_BPeak_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3l.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3m (Bloom Frequency xmap MHW Cum Intensity)
# Loading previously processed causality matrices
MHWCumInt_to_BFreq_CAN = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MHWCumInt/MHWCumInt_to_BFreq_CAN.npy')
MHWCumInt_to_BFreq_NA = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MHWCumInt/MHWCumInt_to_BFreq_NA.npy')
MHWCumInt_to_BFreq_SA = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MHWCumInt/MHWCumInt_to_BFreq_SA.npy')
MHWCumInt_to_BFreq_AL = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MHWCumInt/MHWCumInt_to_BFreq_AL.npy')
MHWCumInt_to_BFreq_BAL = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MHWCumInt/MHWCumInt_to_BFreq_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MHWCumInt_to_BFreq_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MHWCumInt_to_BFreq_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MHWCumInt_to_BFreq_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MHWCumInt_to_BFreq_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BFreq xmap MHW Cum Intensity', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MHWCumInt_to_BFreq_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3m.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3n (Bloom Frequency xmap Windiness)
# Loading previously processed causality matrices
Windiness_to_BFreq_CAN = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_Windiness/Windiness_to_BFreq_CAN.npy')
Windiness_to_BFreq_NA = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_Windiness/Windiness_to_BFreq_NA.npy')
Windiness_to_BFreq_SA = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_Windiness/Windiness_to_BFreq_SA.npy')
Windiness_to_BFreq_AL = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_Windiness/Windiness_to_BFreq_AL.npy')
Windiness_to_BFreq_BAL = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_Windiness/Windiness_to_BFreq_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, Windiness_to_BFreq_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, Windiness_to_BFreq_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, Windiness_to_BFreq_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, Windiness_to_BFreq_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BFreq xmap Windiness', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, Windiness_to_BFreq_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3n.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Supplementary Fig. S3o (Bloom Frequency xmap Max MLD)
# Loading previously processed causality matrices
MaxMLD_to_BFreq_CAN = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MaxMLD/MaxMLD_to_BFreq_CAN.npy')
MaxMLD_to_BFreq_NA = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MaxMLD/MaxMLD_to_BFreq_NA.npy')
MaxMLD_to_BFreq_SA = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MaxMLD/MaxMLD_to_BFreq_SA.npy')
MaxMLD_to_BFreq_AL = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MaxMLD/MaxMLD_to_BFreq_AL.npy')
MaxMLD_to_BFreq_BAL = np.load(r'E:\...\CCM_Outputs\BFreq_xmap_MaxMLD/MaxMLD_to_BFreq_BAL.npy')

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
levels = np.arange(0, 1.1, 0.1)

cs_1= axs.contourf(LON_SA, LAT_SA, MaxMLD_to_BFreq_SA, levels=levels, cmap=cmap, transform=proj)
cs_2= axs.contourf(LON_AL, LAT_AL, MaxMLD_to_BFreq_AL, levels=levels, cmap=cmap, transform=proj)
cs_3= axs.contourf(LON_BAL, LAT_BAL, MaxMLD_to_BFreq_BAL, levels=levels, cmap=cmap, transform=proj)
cs_4= axs.contourf(LON_NA, LAT_NA, MaxMLD_to_BFreq_NA, levels=levels, cmap=cmap, transform=proj)


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
axs.set_title('BFreq xmap Max MLD', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1236, 0.265, 0.265], projection=proj)
# Modify the          [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, MaxMLD_to_BFreq_CAN, levels=levels, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# box_ax.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S3\Fig_S3o.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')

