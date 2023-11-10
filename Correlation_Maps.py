# -*- coding: utf-8 -*-
"""

########################   CHL - MHWs Correlation Maps   ######################

"""

#Loading required libraries
from scipy.io import loadmat
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
from scipy import stats
import matplotlib.path as mpath
import cartopy.feature as cft
from matplotlib import ticker
import matplotlib.colors 
from matplotlib.colors import LinearSegmentedColormap as linearsegm
import xarray as xr
from scipy.stats import spearmanr


##Load MHWs_from_MATLAB.py


###############################################################################
                         ##Pearson Correlation##
def correlation_matrix(X : np.ndarray, Y : np.ndarray) -> np.ndarray:
    X_NAME = 'x'
    Y_NAME = 'y'
    
    corr_matrix = np.zeros(shape = X.shape[:2]) #Shape = (lon, lat)

    for lon in range(X.shape[0]):
        for lat in range(X.shape[1]):
            corr_matrix[lon, lat] = pd.DataFrame({X_NAME : X[lon, lat, :], Y_NAME : Y[lon, lat, :]}).corr()[X_NAME][Y_NAME]
    
    return deepcopy(corr_matrix)
###############################################################################


###############################################################################
                         ##Spearman Correlation##
def correlation_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    corr_matrix = np.zeros(shape=X.shape[:2])  # Shape = (lon, lat)

    for lon in range(X.shape[0]):
        for lat in range(X.shape[1]):
            corr_matrix[lon, lat], _ = spearmanr(X[lon, lat, :], Y[lon, lat, :])

    return corr_matrix
###############################################################################



##Compute Correlation BMHW - SMHW Total Annual Days
Corr_Td_GC = correlation_matrix(BMHW_td_ts_GC_MODEL, MHW_td_ts_GC_MODEL)
Corr_Td_AL = correlation_matrix(BMHW_td_ts_AL_MODEL, MHW_td_ts_AL_MODEL)
Corr_Td_BAL = correlation_matrix(BMHW_td_ts_BAL_MODEL, MHW_td_ts_BAL_MODEL)
Corr_Td_NA = correlation_matrix(BMHW_td_ts_NA_MODEL, MHW_td_ts_NA_MODEL)
Corr_Td_CAN = correlation_matrix(BMHW_td_ts_CAN_MODEL, MHW_td_ts_CAN_MODEL)

##Compute Correlation BMHW - SMHW Maximum Intensity
Corr_Max_GC = correlation_matrix(BMHW_max_ts_GC_MODEL, MHW_max_ts_GC_MODEL)
Corr_Max_AL = correlation_matrix(BMHW_max_ts_AL_MODEL, MHW_max_ts_AL_MODEL)
Corr_Max_BAL = correlation_matrix(BMHW_max_ts_BAL_MODEL, MHW_max_ts_BAL_MODEL)
Corr_Max_NA = correlation_matrix(BMHW_max_ts_NA_MODEL, MHW_max_ts_NA_MODEL)
Corr_Max_CAN = correlation_matrix(BMHW_max_ts_CAN_MODEL, MHW_max_ts_CAN_MODEL)

##Compute Correlation BMHW - SMHW Cumulative Intensity
Corr_Cum_GC = correlation_matrix(BMHW_cum_ts_GC_MODEL, MHW_cum_ts_GC_MODEL)
Corr_Cum_AL = correlation_matrix(BMHW_cum_ts_AL_MODEL, MHW_cum_ts_AL_MODEL)
Corr_Cum_BAL = correlation_matrix(BMHW_cum_ts_BAL_MODEL, MHW_cum_ts_BAL_MODEL)
Corr_Cum_NA = correlation_matrix(BMHW_cum_ts_NA_MODEL, MHW_cum_ts_NA_MODEL)
Corr_Cum_CAN = correlation_matrix(BMHW_cum_ts_CAN_MODEL, MHW_cum_ts_CAN_MODEL)





##Mapplot BMHW - SMHW Total Annual Days    
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.RdBu_r
# cmap=plt.cm.RdYlBu_r

levels=np.linspace(-0.7, 0.7, num=15)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, Corr_Td_GC, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, Corr_Td_AL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, Corr_Td_BAL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, Corr_Td_NA, levels, cmap=cmap, transform=proj, extend ='both')

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
cbar.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
cbar.set_label(r'Spearman coeficient (ρ)', fontsize=22)
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
axs.set_title(r'BMHW - SMHW Total Annual Days', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Corr_Td_CAN, levels, cmap=cmap, transform=proj, extend ='both')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_5\Pearson_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






##Mapplot BMHW - SMHW Maximum Intensity
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.RdBu_r
# cmap=plt.cm.RdYlBu_r

levels=np.linspace(-0.7, 0.7, num=15)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, Corr_Max_GC, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, Corr_Max_AL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, Corr_Max_BAL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, Corr_Max_NA, levels, cmap=cmap, transform=proj, extend ='both')

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
cbar.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
cbar.set_label(r'Spearman coeficient (ρ)', fontsize=22)
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
axs.set_title(r'BMHW - SMHW Maximum Intensity', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Corr_Max_CAN, levels, cmap=cmap, transform=proj, extend ='both')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_5\Pearson_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







##Mapplot BMHW - SMHW Cumulative Intensity
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.RdBu_r
# cmap=plt.cm.RdYlBu_r

levels=np.linspace(-0.7, 0.7, num=15)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, Corr_Cum_GC, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, Corr_Cum_AL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, Corr_Cum_BAL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, Corr_Cum_NA, levels, cmap=cmap, transform=proj, extend ='both')

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
cbar.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
cbar.set_label(r'Pearson coeficient ($r$)', fontsize=22)
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
axs.set_title(r'BMHW - SMHW Cumulative Intensity', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Corr_Max_CAN, levels, cmap=cmap, transform=proj, extend ='both')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_5\Spearman_Cum.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





