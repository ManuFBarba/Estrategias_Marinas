# -*- coding: utf-8 -*-
"""

############################ Map Plot SMD MHWs  ###############################

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

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap


##Loading sst data from Datos_ESMARES.py





                              #################
                              ## Total SMHWs ##
                              #################

##########################
# Total Annual SMHW days #
##########################

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(15, 40, num=26)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, np.nanmean(np.where(MHW_td_ts_GC_MODEL == 0, np.NaN, MHW_td_ts_GC_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, np.nanmean(np.where(MHW_td_ts_AL_MODEL == 0, np.NaN, MHW_td_ts_AL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, np.nanmean(np.where(MHW_td_ts_BAL_MODEL == 0, np.NaN, MHW_td_ts_BAL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, np.nanmean(np.where(MHW_td_ts_NA_MODEL == 0, np.NaN, MHW_td_ts_NA_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')

# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([15, 20, 25, 30, 35, 40])
cbar.set_label(r'[$days$]', fontsize=22)
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
axs.set_title('Total Annual SMHW days', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, np.nanmean(np.where(MHW_td_ts_CAN_MODEL == 0, np.NaN, MHW_td_ts_CAN_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\SMHW_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





####################################################
# Total Annual SMHW days 2008-2022 minus 1993-2007 #
####################################################

td_diff_GC = np.nanmean(np.where(MHW_td_ts_GC_MODEL[:,:,15:30] == 0, np.NaN, MHW_td_ts_GC_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(MHW_td_ts_GC_MODEL[:,:,0:15] == 0, np.NaN, MHW_td_ts_GC_MODEL[:,:,0:15]), axis=2)
signif_td_GC = np.where(td_diff_GC >= 5, 1, np.NaN)
signif_td_GC = np.where(td_diff_GC <= -5, 1, signif_td_GC)

td_diff_AL = np.nanmean(np.where(MHW_td_ts_AL_MODEL[:,:,15:30] == 0, np.NaN, MHW_td_ts_AL_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(MHW_td_ts_AL_MODEL[:,:,0:15] == 0, np.NaN, MHW_td_ts_AL_MODEL[:,:,0:15]), axis=2)
signif_td_AL = np.where(td_diff_AL >= 5, 1, np.NaN)
signif_td_AL = np.where(td_diff_AL <= -5, 1, signif_td_AL)

td_diff_BAL = np.nanmean(np.where(MHW_td_ts_BAL_MODEL[:,:,15:30] == 0, np.NaN, MHW_td_ts_BAL_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(MHW_td_ts_BAL_MODEL[:,:,0:15] == 0, np.NaN, MHW_td_ts_BAL_MODEL[:,:,0:15]), axis=2)
signif_td_BAL = np.where(td_diff_BAL >= 5, 1, np.NaN)
signif_td_BAL = np.where(td_diff_BAL <= -5, 1, signif_td_BAL)

td_diff_NA = np.nanmean(np.where(MHW_td_ts_NA_MODEL[:,:,15:30] == 0, np.NaN, MHW_td_ts_NA_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(MHW_td_ts_NA_MODEL[:,:,0:15] == 0, np.NaN, MHW_td_ts_NA_MODEL[:,:,0:15]), axis=2)
signif_td_NA = np.where(td_diff_NA >= 5, 1, np.NaN)
signif_td_NA = np.where(td_diff_NA <= -5, 1, signif_td_NA)

td_diff_CAN = np.nanmean(np.where(MHW_td_ts_CAN_MODEL[:,:,15:30] == 0, np.NaN, MHW_td_ts_CAN_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(MHW_td_ts_CAN_MODEL[:,:,0:15] == 0, np.NaN, MHW_td_ts_CAN_MODEL[:,:,0:15]), axis=2)
signif_td_CAN = np.where(td_diff_CAN >= 5, 1, np.NaN)
signif_td_CAN = np.where(td_diff_CAN <= -5, 1, signif_td_CAN)


fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
levels=np.linspace(-25, 25, 21)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, td_diff_GC, levels, cmap=cmap, transform=proj, extend ='both')
css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_td_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, td_diff_AL, levels, cmap=cmap, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_td_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, td_diff_BAL, levels, cmap=cmap, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_td_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, td_diff_NA, levels, cmap=cmap, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_td_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25])
cbar.set_label(r'[$days$]', fontsize=22)
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
axs.set_title('Total Annual SMHW days 2008-2022 minus 1993-2007', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, td_diff_CAN, levels, cmap=cmap, transform=proj, extend ='both')
css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_td_CAN[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)

# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)



outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\SMHW_Td_diff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





###########################
#  SMHW Maximum Intensity #
###########################

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(1.25, 3, num=22)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_max_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_max_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_max_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_max_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3])
cbar.set_label(r'[$^\circ$C]', fontsize=22)
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
axs.set_title(r'SMHW Maximum Intensity', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_max_CAN_MODEL+0.25, levels, cmap=cmap, transform=proj, extend ='both')
# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\SMHW_MaxInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




####################################################
# SMHW Maximum Intensity 2008-2022 minus 1993-2007 #
####################################################

max_diff_GC = np.nanmean(MHW_max_ts_GC_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_max_ts_GC_MODEL[:,:,0:15], axis=2)
signif_max_GC = np.where(max_diff_GC >= 0.25, 1, np.NaN)
signif_max_GC = np.where(max_diff_GC <= -0.25, 1, signif_max_GC)

max_diff_AL = np.nanmean(MHW_max_ts_AL_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_max_ts_AL_MODEL[:,:,0:15], axis=2)
signif_max_AL = np.where(max_diff_AL >= 0.25, 1, np.NaN)
signif_max_AL = np.where(max_diff_AL <= -0.25, 1, signif_max_AL)

max_diff_BAL = np.nanmean(MHW_max_ts_BAL_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_max_ts_BAL_MODEL[:,:,0:15], axis=2)
signif_max_BAL = np.where(max_diff_BAL >= 0.15, 1, np.NaN)
signif_max_BAL = np.where(max_diff_BAL <= -0.15, 1, signif_max_BAL)

max_diff_NA = np.nanmean(MHW_max_ts_NA_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_max_ts_NA_MODEL[:,:,0:15], axis=2)
signif_max_NA = np.where(max_diff_NA >= 0.15, 1, np.NaN)
signif_max_NA = np.where(max_diff_NA <= -0.15, 1, signif_max_NA)

max_diff_CAN = np.nanmean(MHW_max_ts_CAN_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_max_ts_CAN_MODEL[:,:,0:15], axis=2)
signif_max_CAN = np.where(max_diff_CAN >= 0.05, 1, np.NaN)
signif_max_CAN = np.where(max_diff_CAN <= -0.05, 1, signif_max_CAN)


fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
levels=np.linspace(-1, 1, 25)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, max_diff_GC, levels, cmap=cmap, transform=proj, extend ='both')
css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_max_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, max_diff_AL, levels, cmap=cmap, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_max_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, max_diff_BAL, levels, cmap=cmap, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_max_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, max_diff_NA, levels, cmap=cmap, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_max_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
cbar.set_label(r'[$^\circ$C]', fontsize=22)
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
axs.set_title('SMHW Maximum Intensity 2008-2022 minus 1993-2007', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, max_diff_CAN, levels, cmap=cmap, transform=proj, extend ='both')
css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_max_CAN[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)

# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)



outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\SMHW_max_diff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




##############################
#  SMHW Cumulative Intensity #
##############################

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(10, 35, num=26)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_cum_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_cum_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_cum_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_cum_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([10, 15, 20, 25, 30, 35])
cbar.set_label(r'[$^{\circ}C\ ·  days$]', fontsize=22)
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
axs.set_title(r'SMHW Cumulative Intensity', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_cum_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\SMHW_CumInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




#######################################################
# SMHW Cumulative Intensity 2008-2022 minus 1993-2007 #
#######################################################

max_diff_GC = np.nanmean(MHW_cum_ts_GC_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_cum_ts_GC_MODEL[:,:,0:15], axis=2)
signif_cum_GC = np.where(max_diff_GC >= 5, 1, np.NaN)
signif_cum_GC = np.where(max_diff_GC <= -5, 1, signif_cum_GC)

max_diff_AL = np.nanmean(MHW_cum_ts_AL_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_cum_ts_AL_MODEL[:,:,0:15], axis=2)
signif_cum_AL = np.where(max_diff_AL >= 5, 1, np.NaN)
signif_cum_AL = np.where(max_diff_AL <= -5, 1, signif_cum_AL)

max_diff_BAL = np.nanmean(MHW_cum_ts_BAL_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_cum_ts_BAL_MODEL[:,:,0:15], axis=2)
signif_cum_BAL = np.where(max_diff_BAL >= 5, 1, np.NaN)
signif_cum_BAL = np.where(max_diff_BAL <= -5, 1, signif_cum_BAL)

max_diff_NA = np.nanmean(MHW_cum_ts_NA_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_cum_ts_NA_MODEL[:,:,0:15], axis=2)
signif_cum_NA = np.where(max_diff_NA >= 2.5, 1, np.NaN)
signif_cum_NA = np.where(max_diff_NA <= -2.5, 1, signif_cum_NA)

max_diff_CAN = np.nanmean(MHW_cum_ts_CAN_MODEL[:,:,15:30], axis=2) - np.nanmean(MHW_cum_ts_CAN_MODEL[:,:,0:15], axis=2)
signif_cum_CAN = np.where(max_diff_CAN >= 5, 1, np.NaN)
signif_cum_CAN = np.where(max_diff_CAN <= -5, 1, signif_cum_CAN)


fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
levels=np.linspace(-20, 20, 25)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, max_diff_GC, levels, cmap=cmap, transform=proj, extend ='both')
css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_cum_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, max_diff_AL, levels, cmap=cmap, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_cum_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, max_diff_BAL, levels, cmap=cmap, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_cum_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, max_diff_NA, levels, cmap=cmap, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_cum_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
cbar.set_label(r'[$^{\circ}C\ ·  days$]', fontsize=22)
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
axs.set_title('SMHW Cumulative Intensity 2008-2022 minus 1993-2007', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, max_diff_CAN, levels, cmap=cmap, transform=proj, extend ='both')
css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_cum_CAN[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)

# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\SMHW_Cum_diff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



























                              #################
                              ## Total BMHWs ##
                              #################

##########################
# Total Annual BMHW days #
##########################

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(20, 120, num=21)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, np.nanmean(np.where(BMHW_td_ts_GC_MODEL == 0, np.NaN, BMHW_td_ts_GC_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, np.nanmean(np.where(BMHW_td_ts_AL_MODEL == 0, np.NaN, BMHW_td_ts_AL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, np.nanmean(np.where(BMHW_td_ts_BAL_MODEL == 0, np.NaN, BMHW_td_ts_BAL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, np.nanmean(np.where(BMHW_td_ts_NA_MODEL == 0, np.NaN, BMHW_td_ts_NA_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')

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


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([20, 40, 60, 80, 100, 120])
cbar.set_label(r'[$days$]', fontsize=22)
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
axs.set_title('Total Annual BMHW days', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, np.nanmean(np.where(BMHW_td_ts_CAN_MODEL == 0, np.NaN, BMHW_td_ts_CAN_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\BMHW_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





####################################################
# Total Annual BMHW days 2008-2022 minus 1993-2007 #
####################################################

# td_diff_GC = np.nanmean(np.where(BMHW_td_ts_GC_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_GC_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_GC_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_GC_MODEL[:,:,0:15]), axis=2)
# signif_td_GC = np.where(td_diff_GC >= 5, 1, np.NaN)
# signif_td_GC = np.where(td_diff_GC <= -5, 1, signif_td_GC)

# td_diff_AL = np.nanmean(np.where(BMHW_td_ts_AL_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_AL_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_AL_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_AL_MODEL[:,:,0:15]), axis=2)
# signif_td_AL = np.where(td_diff_AL >= 5, 1, np.NaN)
# signif_td_AL = np.where(td_diff_AL <= -5, 1, signif_td_AL)

# td_diff_BAL = np.nanmean(np.where(BMHW_td_ts_BAL_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_BAL_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_BAL_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_BAL_MODEL[:,:,0:15]), axis=2)
# signif_td_BAL = np.where(td_diff_BAL >= 5, 1, np.NaN)
# signif_td_BAL = np.where(td_diff_BAL <= -5, 1, signif_td_BAL)

# td_diff_NA = np.nanmean(np.where(BMHW_td_ts_NA_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_NA_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_NA_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_NA_MODEL[:,:,0:15]), axis=2)
# signif_td_NA = np.where(td_diff_NA >= 5, 1, np.NaN)
# signif_td_NA = np.where(td_diff_NA <= -5, 1, signif_td_NA)

# td_diff_CAN = np.nanmean(np.where(BMHW_td_ts_CAN_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_CAN_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_CAN_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_CAN_MODEL[:,:,0:15]), axis=2)
# signif_td_CAN = np.where(td_diff_CAN >= 5, 1, np.NaN)
# signif_td_CAN = np.where(td_diff_CAN <= -5, 1, signif_td_CAN)


# fig = plt.figure(figsize=(20, 10))

# proj = ccrs.PlateCarree()  # Choose the projection
# # Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# axs = plt.axes(projection=proj)
# # Set tick font size
# for label in (axs.get_xticklabels() + axs.get_yticklabels()):
#     label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# levels=np.linspace(-40, 40, 25)

# cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, td_diff_GC, levels, cmap=cmap, transform=proj, extend ='both')
# css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_td_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
# cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, td_diff_AL, levels, cmap=cmap, transform=proj, extend ='both')
# css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_td_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
# cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, td_diff_BAL, levels, cmap=cmap, transform=proj, extend ='both')
# css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_td_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
# cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, td_diff_NA, levels, cmap=cmap, transform=proj, extend ='both')
# css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_td_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# # # Define contour levels for elevation and plot elevation isolines
# # elev_levels = [400, 2500]

# # el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
# #                   linestyles=['solid', 'dashed'], linewidths=1.25)
# # for line_1 in el_1.collections:
# #     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# # el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
# #                   linestyles=['solid', 'dashed'], linewidths=1.25)
# # for line_2 in el_2.collections:
# #     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# # el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
# #                   linestyles=['solid', 'dashed'], linewidths=1.25)
# # for line_3 in el_3.collections:
# #     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# # el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
# #                   linestyles=['solid', 'dashed'], linewidths=1.25)
# # for line_4 in el_4.collections:
# #     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


# cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
# cbar.ax.minorticks_off()
# # cbar.set_ticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25])
# cbar.set_label(r'[$days$]', fontsize=22)
# # Add map features
# axs.coastlines(resolution='10m', color='black', linewidth=1)
# axs.add_feature(land_10m)

# # Set the extent of the main plot
# axs.set_extent([-16, 6.5, 34, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# axs.xaxis.set_major_formatter(lon_formatter)
# axs.yaxis.set_major_formatter(lat_formatter)
# axs.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
# axs.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)
# #Set the title
# axs.set_title('Total Annual BMHW days 2008-2022 minus 1993-2007', fontsize=25)

# # Create a second set of axes (subplots) for the Canarian box
# box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# # Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# # Plot the data for Canarias on the second axes
# cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, td_diff_CAN, levels, cmap=cmap, transform=proj, extend ='both')
# css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_td_CAN[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)

# # el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
# #                       linestyles=['solid', 'dashed'], linewidths=1.25)
# # for line_5 in el_5.collections:
# #     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# # Add map features for the second axes
# box_ax.coastlines(resolution='10m', color='black', linewidth=1)
# box_ax.add_feature(land_10m)



# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\BMHW_Td_diff.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





###############################################################################
signif_td_GC = BMHW_td_dtr_GC_MODEL
signif_td_GC = np.where(signif_td_GC >= 0.1, np.NaN, signif_td_GC)
signif_td_GC = np.where(signif_td_GC < 0.1, 1, signif_td_GC)

signif_td_AL = BMHW_td_dtr_AL_MODEL
signif_td_AL = np.where(signif_td_AL >= 0.1, np.NaN, signif_td_AL)
signif_td_AL = np.where(signif_td_AL < 0.1, 1, signif_td_AL)

signif_td_BAL = BMHW_td_dtr_BAL_MODEL
signif_td_BAL = np.where(signif_td_BAL >= 0.1, np.NaN, signif_td_BAL)
signif_td_BAL = np.where(signif_td_BAL < 0.1, 1, signif_td_BAL)

signif_td_NA = BMHW_td_dtr_NA_MODEL
signif_td_NA = np.where(signif_td_NA >= 0.1, np.NaN, signif_td_NA)
signif_td_NA = np.where(signif_td_NA < 0.1, 1, signif_td_NA)

signif_td_canarias = BMHW_td_dtr_CAN_MODEL
signif_td_canarias = np.where(signif_td_canarias >= 0.1, np.NaN, signif_td_canarias)
signif_td_canarias = np.where(signif_td_canarias < 0.1, 1, signif_td_canarias)

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
levels=np.linspace(-35, 35, 29)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_td_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_td_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_td_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_td_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_td_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_td_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_td_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_td_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


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


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([-30, -20, -10, 0, 10, 20, 30])
cbar.set_label(r'[$days$]', fontsize=22)
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
axs.set_title('Total Annual BMHW days 2008-2022 minus 1993-2007', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_td_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_td_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)



outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\BMHW_Td_diff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)
###############################################################################








###########################
#  BMHW Maximum Intensity #
###########################

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(0, 2.5, num=26)

# Create the colormap with two parts: light gray and YlOrRd
cmap_custom = plt.cm.YlOrRd

# # Define a custom colormap with a gradient from light to dark gray
# gray_colors = np.linspace(0.85, 0.3, 3)
# gray_colormap = LinearSegmentedColormap.from_list('custom_gray', [(c, c, c) for c in gray_colors], N=4)

# # Create a custom colormap with gray gradient for values below 0.3 and YlOrRd for the rest
# colors_custom = gray_colormap(np.linspace(0, 1, 4)).tolist() + plt.cm.YlOrRd(np.linspace(0.1, 1, 23)).tolist()
# cmap_custom = LinearSegmentedColormap.from_list('custom_colormap', colors_custom, N=26)


cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL,  BMHW_max_GC_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_max_AL_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_max_BAL_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_max_NA_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')

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
cbar.set_ticks([0, 0.5, 1, 1.5, 2, 2.5])
cbar.set_label(r'[$^\circ$C]', fontsize=22)
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
axs.set_title(r'BMHW Maximum Intensity', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_max_CAN_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_4\BMHW_MaxInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




####################################################
# BMHW Maximum Intensity 2008-2022 minus 1993-2007 #
####################################################

# max_diff_GC = np.nanmean(BMHW_max_ts_GC_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_GC_MODEL[:,:,0:15], axis=2)
# signif_max_GC = np.where(max_diff_GC >= 0.1, 1, np.NaN)
# signif_max_GC = np.where(max_diff_GC <= -0.1, 1, signif_max_GC)

# max_diff_AL = np.nanmean(BMHW_max_ts_AL_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_AL_MODEL[:,:,0:15], axis=2)
# signif_max_AL = np.where(max_diff_AL >= 0.1, 1, np.NaN)
# signif_max_AL = np.where(max_diff_AL <= -0.1, 1, signif_max_AL)

# max_diff_BAL = np.nanmean(BMHW_max_ts_BAL_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_BAL_MODEL[:,:,0:15], axis=2)
# signif_max_BAL = np.where(max_diff_BAL >= 0.1, 1, np.NaN)
# signif_max_BAL = np.where(max_diff_BAL <= -0.1, 1, signif_max_BAL)

# max_diff_NA = np.nanmean(BMHW_max_ts_NA_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_NA_MODEL[:,:,0:15], axis=2)
# signif_max_NA = np.where(max_diff_NA >= 0.05, 1, np.NaN)
# signif_max_NA = np.where(max_diff_NA <= -0.05, 1, signif_max_NA)

# max_diff_CAN = np.nanmean(BMHW_max_ts_CAN_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_CAN_MODEL[:,:,0:15], axis=2)
# signif_max_CAN = np.where(max_diff_CAN >= 0.05, 1, np.NaN)
# signif_max_CAN = np.where(max_diff_CAN <= -0.05, 1, signif_max_CAN)


# fig = plt.figure(figsize=(20, 10))

# proj = ccrs.PlateCarree()  # Choose the projection
# # Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# axs = plt.axes(projection=proj)
# # Set tick font size
# for label in (axs.get_xticklabels() + axs.get_yticklabels()):
#     label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# levels=np.linspace(-0.4, 0.4, 25)

# cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, max_diff_GC, levels, cmap=cmap, transform=proj, extend ='both')
# css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_max_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
# cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, max_diff_AL, levels, cmap=cmap, transform=proj, extend ='both')
# css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_max_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
# cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, max_diff_BAL, levels, cmap=cmap, transform=proj, extend ='both')
# css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_max_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
# cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, max_diff_NA, levels, cmap=cmap, transform=proj, extend ='both')
# css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_max_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


# cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
# cbar.ax.minorticks_off()
# # cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
# cbar.set_label(r'[$^\circ$C]', fontsize=22)
# # Add map features
# axs.coastlines(resolution='10m', color='black', linewidth=1)
# axs.add_feature(land_10m)

# # Set the extent of the main plot
# axs.set_extent([-16, 6.5, 34, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# axs.xaxis.set_major_formatter(lon_formatter)
# axs.yaxis.set_major_formatter(lat_formatter)
# axs.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
# axs.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)
# #Set the title
# axs.set_title(r'BMHW Maximum Intensity 2008-2022 minus 1993-2007', fontsize=25)

# # Create a second set of axes (subplots) for the Canarian box
# box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# # Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# # Plot the data for Canarias on the second axes
# cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, max_diff_CAN, levels, cmap=cmap, transform=proj, extend ='both')
# css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_max_CAN[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)

# # el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
# #                       linestyles=['solid', 'dashed'], linewidths=1.25)
# # for line_5 in el_5.collections:
# #     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# # Add map features for the second axes
# box_ax.coastlines(resolution='10m', color='black', linewidth=1)
# box_ax.add_feature(land_10m)



# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\SMHW_max_diff.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





###############################################################################
signif_max_GC = np.where(BMHW_max_tr_GC_MODEL*10 >= 0.05, 1, np.NaN)
signif_max_GC = np.where(BMHW_max_tr_GC_MODEL*10 <= -0.05, 1, signif_max_GC)

signif_max_AL = np.where(BMHW_max_tr_AL_MODEL*10 >= 0.05, 1, np.NaN)
signif_max_AL = np.where(BMHW_max_tr_AL_MODEL*10 <= -0.05, 1, signif_max_AL)

signif_max_BAL = np.where(BMHW_max_tr_BAL_MODEL*10 >= 0.05, 1, np.NaN)
signif_max_BAL = np.where(BMHW_max_tr_BAL_MODEL*10 <= -0.05, 1, signif_max_BAL)

signif_max_NA = np.where(BMHW_max_tr_NA_MODEL*10 >= 0.05, 1, np.NaN)
signif_max_NA = np.where(BMHW_max_tr_NA_MODEL*10 <= -0.05, 1, signif_max_NA)

signif_max_CAN = np.where(BMHW_max_tr_CAN_MODEL*10 >= 0.1, 1, np.NaN)
signif_max_CAN = np.where(BMHW_max_tr_CAN_MODEL*10 <= -0.1, 1, signif_max_CAN)

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
levels=np.linspace(-0.4, 0.4, 25)

cmap_custom = plt.cm.RdYlBu_r

# Define a custom colormap with a gradient of grays from -0.05 to 0.05
# gray_colors = np.linspace(0.9, 0.6, 5)
# gray_colormap = LinearSegmentedColormap.from_list('custom_gray', [(c, c, c) for c in gray_colors], N=5)

# # Combine the custom gray colormap with RdYlBu_r for the rest of the values
# colors_custom = plt.cm.Blues_r(np.linspace(0.1, 1, 20)).tolist() + gray_colormap(np.linspace(0, 1, 5)).tolist() + plt.cm.YlOrRd(np.linspace(0.1, 1, 20)).tolist()
# cmap_custom = LinearSegmentedColormap.from_list('custom_colormap', colors_custom, N=25)



cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_max_tr_GC_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_max_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_max_tr_AL_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_max_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_max_tr_BAL_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_max_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_max_tr_NA_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_max_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


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


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
# cbar.set_ticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])
cbar.set_label(r'[$^\circ$C]', fontsize=22)
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
axs.set_title(r'BMHW Maximum Intensity 2008-2022 minus 1993-2007', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_max_tr_CAN_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_max_CAN[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)



outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_4\BMHW_max_diff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)
###############################################################################












##############################
#  BMHW Cumulative Intensity #
##############################

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(0, 50, num=25)

# Create the colormap with two parts: light gray and YlOrRd
cmap_custom = plt.cm.YlOrRd

# # Define a custom colormap with a gradient from light to dark gray
# gray_colors = np.linspace(0.85, 0.3, 4)
# gray_colormap = LinearSegmentedColormap.from_list('custom_gray', [(c, c, c) for c in gray_colors], N=4)

# # Create a custom colormap with gray gradient for values below 0.3 and YlOrRd for the rest
# colors_custom = gray_colormap(np.linspace(0, 1, 4)).tolist() + plt.cm.YlOrRd(np.linspace(0.1, 1, 23)).tolist()
# cmap_custom = LinearSegmentedColormap.from_list('custom_colormap', colors_custom, N=26)


cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL,  BMHW_cum_GC_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_cum_AL_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_cum_BAL_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_cum_NA_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')

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


cbar = plt.colorbar(cs_1, ax=axs, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 10, 20, 30, 40, 50])
cbar.set_label(r'[$^{\circ}C\ ·  days$]', fontsize=22)
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
axs.set_title(r'BMHW Cumulative Intensity', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_cum_CAN_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_4\BMHW_CumInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





#######################################################
# BMHW Cumulative Intensity 2008-2022 minus 1993-2007 #
#######################################################

# max_diff_GC = np.nanmean(BMHW_cum_ts_GC_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_GC_MODEL[:,:,0:15], axis=2)
# signif_cum_GC = np.where(max_diff_GC >= 5, 1, np.NaN)
# signif_cum_GC = np.where(max_diff_GC <= -5, 1, signif_cum_GC)

# max_diff_AL = np.nanmean(BMHW_cum_ts_AL_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_AL_MODEL[:,:,0:15], axis=2)
# signif_cum_AL = np.where(max_diff_AL >= 5, 1, np.NaN)
# signif_cum_AL = np.where(max_diff_AL <= -5, 1, signif_cum_AL)

# max_diff_BAL = np.nanmean(BMHW_cum_ts_BAL_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_BAL_MODEL[:,:,0:15], axis=2)
# signif_cum_BAL = np.where(max_diff_BAL >= 5, 1, np.NaN)
# signif_cum_BAL = np.where(max_diff_BAL <= -5, 1, signif_cum_BAL)

# max_diff_NA = np.nanmean(BMHW_cum_ts_NA_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_NA_MODEL[:,:,0:15], axis=2)
# signif_cum_NA = np.where(max_diff_NA >= 2.5, 1, np.NaN)
# signif_cum_NA = np.where(max_diff_NA <= -2.5, 1, signif_cum_NA)

# max_diff_CAN = np.nanmean(BMHW_cum_ts_CAN_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_CAN_MODEL[:,:,0:15], axis=2)
# signif_cum_CAN = np.where(max_diff_CAN >= 5, 1, np.NaN)
# signif_cum_CAN = np.where(max_diff_CAN <= -5, 1, signif_cum_CAN)


# fig = plt.figure(figsize=(20, 10))

# proj = ccrs.PlateCarree()  # Choose the projection
# # Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# axs = plt.axes(projection=proj)
# # Set tick font size
# for label in (axs.get_xticklabels() + axs.get_yticklabels()):
#     label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# levels=np.linspace(-20, 20, 25)

# cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, max_diff_GC, levels, cmap=cmap, transform=proj, extend ='both')
# css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_cum_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
# cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, max_diff_AL, levels, cmap=cmap, transform=proj, extend ='both')
# css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_cum_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
# cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, max_diff_BAL, levels, cmap=cmap, transform=proj, extend ='both')
# css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_cum_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
# cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, max_diff_NA, levels, cmap=cmap, transform=proj, extend ='both')
# css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_cum_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = axs.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


# cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
# cbar.ax.minorticks_off()
# cbar.set_ticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
# cbar.set_label(r'[$^{\circ}C\ ·  days$]', fontsize=22)
# # Add map features
# axs.coastlines(resolution='10m', color='black', linewidth=1)
# axs.add_feature(land_10m)

# # Set the extent of the main plot
# axs.set_extent([-16, 6.5, 34, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# axs.xaxis.set_major_formatter(lon_formatter)
# axs.yaxis.set_major_formatter(lat_formatter)
# axs.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
# axs.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)
# #Set the title
# axs.set_title(r'BMHW Cumulative Intensity 2008-2022 minus 1993-2007', fontsize=25)

# # Create a second set of axes (subplots) for the Canarian box
# box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# # Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# # Plot the data for Canarias on the second axes
# cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, max_diff_CAN, levels, cmap=cmap, transform=proj, extend ='both')
# css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_cum_CAN[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)

# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# # Add map features for the second axes
# box_ax.coastlines(resolution='10m', color='black', linewidth=1)
# box_ax.add_feature(land_10m)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_3\SMHW_Cum_diff.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




###############################################################################
signif_cum_GC = np.where(BMHW_cum_tr_GC_MODEL*10 >= 5, 1, np.NaN)
signif_cum_GC = np.where(BMHW_cum_tr_GC_MODEL*10 <= -5, 1, signif_cum_GC)

signif_cum_AL = np.where(BMHW_cum_tr_AL_MODEL*10 >= 5, 1, np.NaN)
signif_cum_AL = np.where(BMHW_cum_tr_AL_MODEL*10 <= -5, 1, signif_cum_AL)

signif_cum_BAL = np.where(BMHW_cum_tr_BAL_MODEL*10 >= 5, 1, np.NaN)
signif_cum_BAL = np.where(BMHW_cum_tr_BAL_MODEL*10 <= -5, 1, signif_cum_BAL)

signif_cum_NA = np.where(BMHW_cum_tr_NA_MODEL*10 >= 5, 1, np.NaN)
signif_cum_NA = np.where(BMHW_cum_tr_NA_MODEL*10 <= -5, 1, signif_cum_NA)

signif_cum_CAN = np.where(BMHW_cum_tr_CAN_MODEL*10 >= 2.5, 1, np.NaN)
signif_cum_CAN = np.where(BMHW_cum_tr_CAN_MODEL*10 <= -2.5, 1, signif_cum_CAN)

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
levels=np.linspace(-20, 20, 25)

cmap_custom = plt.cm.RdYlBu_r

# # Define a custom colormap with a gradient of grays from -0.05 to 0.05
# gray_colors = np.linspace(0.9, 0.6, 5)
# gray_colormap = LinearSegmentedColormap.from_list('custom_gray', [(c, c, c) for c in gray_colors], N=5)

# # Combine the custom gray colormap with RdYlBu_r for the rest of the values
# colors_custom = plt.cm.Blues_r(np.linspace(0.1, 1, 20)).tolist() + gray_colormap(np.linspace(0, 1, 5)).tolist() + plt.cm.YlOrRd(np.linspace(0.1, 1, 20)).tolist()
# cmap_custom = LinearSegmentedColormap.from_list('custom_colormap', colors_custom, N=25)



cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_cum_tr_GC_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_cum_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_cum_tr_AL_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_cum_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_cum_tr_BAL_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_cum_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_cum_tr_NA_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_cum_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


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


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
# cbar.set_ticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])
cbar.set_label(r'[$^{\circ}C\ ·  days$]', fontsize=22)
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
axs.set_title(r'BMHW Cumulative Intensity 2008-2022 minus 1993-2007', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_cum_tr_CAN_MODEL*10, levels, cmap=cmap_custom, transform=proj, extend ='both')
css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_cum_CAN[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)



outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_4\BMHW_cum_diff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)
###############################################################################












                             #######################
                             ## Total BMHW - SMHW ##
                             #######################
                             
                             
## Total Annual MHW Days      
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

levels=np.linspace(-10, 10, num=21)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_td_GC_MODEL - MHW_td_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_td_AL_MODEL - MHW_td_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_td_BAL_MODEL - MHW_td_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_td_NA_MODEL - MHW_td_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

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


cbar = plt.colorbar(cs_1, ax=axs, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
cbar.set_label(r'[$days$]', fontsize=22)
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
axs.set_title(r'Total Annual BMHW - SMHW Days', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_td_CAN_MODEL - MHW_td_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_5\BMHW_SMHW_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



               


## Maximum Intensity   
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

levels=np.linspace(-2, 2, num=17)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_max_GC_MODEL - MHW_max_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_max_AL_MODEL - MHW_max_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_max_BAL_MODEL - MHW_max_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_max_NA_MODEL - MHW_max_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

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
cbar.set_ticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
cbar.set_label(r'[$^\circ$C]', fontsize=22)
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
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_max_CAN_MODEL - MHW_max_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_5\BMHW_SMHW_MaxInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







    
                  
## Cumulative Intensity         
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

levels=np.linspace(-30, 30, num=25)

cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_cum_GC_MODEL - MHW_cum_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_cum_AL_MODEL - MHW_cum_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_cum_BAL_MODEL - MHW_cum_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_cum_NA_MODEL - MHW_cum_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

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


cbar = plt.colorbar(cs_1, ax=axs, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([-30, -20, -10, 0, 10, 20, 30])
cbar.set_label(r'[$^{\circ}C\ ·  days$]', fontsize=22)
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
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_cum_CAN_MODEL - MHW_cum_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_5\BMHW_SMHW_CumInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










