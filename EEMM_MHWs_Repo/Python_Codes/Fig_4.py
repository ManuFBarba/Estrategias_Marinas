# -*- coding: utf-8 -*-
"""

#########################      Figure 4 in 
Fernández-Barba, M., Huertas, I. E., & Navarro, G. (2024). 
Assessment of surface and bottom marine heatwaves along the Spanish coast. 
Ocean Modelling, 190, 102399.                          ########################

"""

#Loading required python modules
import numpy as np

import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cft


##Load previously-clipped bethymetry datasets (Datos_ESMARES.py)
##Load MHWs_from_MATLAB.py




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

cs_1= axs.contourf(LON_SA_MODEL, LAT_SA_MODEL, np.nanmean(np.where(BMHW_td_ts_SA_MODEL == 0, np.NaN, BMHW_td_ts_SA_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, np.nanmean(np.where(BMHW_td_ts_AL_MODEL == 0, np.NaN, BMHW_td_ts_AL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, np.nanmean(np.where(BMHW_td_ts_BAL_MODEL == 0, np.NaN, BMHW_td_ts_BAL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, np.nanmean(np.where(BMHW_td_ts_NA_MODEL == 0, np.NaN, BMHW_td_ts_NA_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['black', 'black'], transform=proj,
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


outfile = r'...\Fig_4\4a_BMHW_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




####################################################
# Total Annual BMHW days 2008-2022 minus 1993-2007 #
####################################################

td_diff_SA = np.nanmean(np.where(BMHW_td_ts_SA_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_SA_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_SA_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_SA_MODEL[:,:,0:15]), axis=2)
signif_td_SA = np.where(td_diff_SA >= 5, 1, np.NaN)
signif_td_SA = np.where(td_diff_SA <= -5, 1, signif_td_SA)

td_diff_AL = np.nanmean(np.where(BMHW_td_ts_AL_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_AL_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_AL_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_AL_MODEL[:,:,0:15]), axis=2)
signif_td_AL = np.where(td_diff_AL >= 5, 1, np.NaN)
signif_td_AL = np.where(td_diff_AL <= -5, 1, signif_td_AL)

td_diff_BAL = np.nanmean(np.where(BMHW_td_ts_BAL_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_BAL_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_BAL_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_BAL_MODEL[:,:,0:15]), axis=2)
signif_td_BAL = np.where(td_diff_BAL >= 5, 1, np.NaN)
signif_td_BAL = np.where(td_diff_BAL <= -5, 1, signif_td_BAL)

td_diff_NA = np.nanmean(np.where(BMHW_td_ts_NA_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_NA_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_NA_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_NA_MODEL[:,:,0:15]), axis=2)
signif_td_NA = np.where(td_diff_NA >= 5, 1, np.NaN)
signif_td_NA = np.where(td_diff_NA <= -5, 1, signif_td_NA)

td_diff_CAN = np.nanmean(np.where(BMHW_td_ts_CAN_MODEL[:,:,15:30] == 0, np.NaN, BMHW_td_ts_CAN_MODEL[:,:,15:30]), axis=2) - np.nanmean(np.where(BMHW_td_ts_CAN_MODEL[:,:,0:15] == 0, np.NaN, BMHW_td_ts_CAN_MODEL[:,:,0:15]), axis=2)
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
levels=np.linspace(-35, 35, 29)

cs_1= axs.contourf(LON_SA_MODEL, LAT_SA_MODEL, td_diff_SA, levels, cmap=cmap, transform=proj, extend ='both')
css_1=axs.scatter(LON_SA_MODEL[::8,::8], LAT_SA_MODEL[::8,::8], signif_td_SA[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, td_diff_AL, levels, cmap=cmap, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_td_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, td_diff_BAL, levels, cmap=cmap, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_td_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, td_diff_NA, levels, cmap=cmap, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_td_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['black', 'black'], transform=proj,
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
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, td_diff_CAN, levels, cmap=cmap, transform=proj, extend ='both')
css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_td_CAN[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)



outfile = r'...\Fig_4\4b_BMHW_Td_diff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




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

levels=np.linspace(0, 2.5, num=26)

# Create the colormap with two parts: light gray and YlOrRd
cmap_custom = plt.cm.YlOrRd


cs_1= axs.contourf(LON_SA_MODEL, LAT_SA_MODEL,  BMHW_max_SA_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_max_AL_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_max_BAL_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_max_NA_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['black', 'black'], transform=proj,
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


outfile = r'...\Fig_4\4c_BMHW_MaxInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




####################################################
# SMHW Maximum Intensity 2008-2022 minus 1993-2007 #
####################################################

max_diff_SA = np.nanmean(BMHW_max_ts_SA_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_SA_MODEL[:,:,0:15], axis=2)
signif_max_SA = np.where(max_diff_SA >= 0.1, 1, np.NaN)
signif_max_SA = np.where(max_diff_SA <= -0.1, 1, signif_max_SA)

max_diff_AL = np.nanmean(BMHW_max_ts_AL_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_AL_MODEL[:,:,0:15], axis=2)
signif_max_AL = np.where(max_diff_AL >= 0.1, 1, np.NaN)
signif_max_AL = np.where(max_diff_AL <= -0.1, 1, signif_max_AL)

max_diff_BAL = np.nanmean(BMHW_max_ts_BAL_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_BAL_MODEL[:,:,0:15], axis=2)
signif_max_BAL = np.where(max_diff_BAL >= 0.1, 1, np.NaN)
signif_max_BAL = np.where(max_diff_BAL <= -0.1, 1, signif_max_BAL)

max_diff_NA = np.nanmean(BMHW_max_ts_NA_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_NA_MODEL[:,:,0:15], axis=2)
signif_max_NA = np.where(max_diff_NA >= 0.05, 1, np.NaN)
signif_max_NA = np.where(max_diff_NA <= -0.05, 1, signif_max_NA)

max_diff_CAN = np.nanmean(BMHW_max_ts_CAN_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_max_ts_CAN_MODEL[:,:,0:15], axis=2)
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
levels=np.linspace(-0.4, 0.4, 25)

cs_1= axs.contourf(LON_SA_MODEL, LAT_SA_MODEL, max_diff_SA, levels, cmap=cmap, transform=proj, extend ='both')
css_1=axs.scatter(LON_SA_MODEL[::8,::8], LAT_SA_MODEL[::8,::8], signif_max_SA[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, max_diff_AL, levels, cmap=cmap, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_max_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, max_diff_BAL, levels, cmap=cmap, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_max_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, max_diff_NA, levels, cmap=cmap, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_max_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['black', 'black'], transform=proj,
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
# cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
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
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, max_diff_CAN, levels, cmap=cmap, transform=proj, extend ='both')
css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_max_CAN[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'...\Fig_4\4d_BMHW_max_diff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




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

levels=np.linspace(0, 50, num=25)

cmap_custom = plt.cm.YlOrRd


cs_1= axs.contourf(LON_SA_MODEL, LAT_SA_MODEL,  BMHW_cum_SA_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_cum_AL_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_cum_BAL_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_cum_NA_MODEL, levels, cmap=cmap_custom, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['black', 'black'], transform=proj,
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


outfile = r'...\Fig_4\4e_BMHW_CumInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




#######################################################
# SMHW Cumulative Intensity 2008-2022 minus 1993-2007 #
#######################################################

max_diff_SA = np.nanmean(BMHW_cum_ts_SA_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_SA_MODEL[:,:,0:15], axis=2)
signif_cum_SA = np.where(max_diff_SA >= 5, 1, np.NaN)
signif_cum_SA = np.where(max_diff_SA <= -5, 1, signif_cum_SA)

max_diff_AL = np.nanmean(BMHW_cum_ts_AL_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_AL_MODEL[:,:,0:15], axis=2)
signif_cum_AL = np.where(max_diff_AL >= 5, 1, np.NaN)
signif_cum_AL = np.where(max_diff_AL <= -5, 1, signif_cum_AL)

max_diff_BAL = np.nanmean(BMHW_cum_ts_BAL_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_BAL_MODEL[:,:,0:15], axis=2)
signif_cum_BAL = np.where(max_diff_BAL >= 5, 1, np.NaN)
signif_cum_BAL = np.where(max_diff_BAL <= -5, 1, signif_cum_BAL)

max_diff_NA = np.nanmean(BMHW_cum_ts_NA_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_NA_MODEL[:,:,0:15], axis=2)
signif_cum_NA = np.where(max_diff_NA >= 2.5, 1, np.NaN)
signif_cum_NA = np.where(max_diff_NA <= -2.5, 1, signif_cum_NA)

max_diff_CAN = np.nanmean(BMHW_cum_ts_CAN_MODEL[:,:,15:30], axis=2) - np.nanmean(BMHW_cum_ts_CAN_MODEL[:,:,0:15], axis=2)
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

cs_1= axs.contourf(LON_SA_MODEL, LAT_SA_MODEL, max_diff_SA, levels, cmap=cmap, transform=proj, extend ='both')
css_1=axs.scatter(LON_SA_MODEL[::8,::8], LAT_SA_MODEL[::8,::8], signif_cum_SA[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, max_diff_AL, levels, cmap=cmap, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_cum_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, max_diff_BAL, levels, cmap=cmap, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_cum_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, max_diff_NA, levels, cmap=cmap, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_cum_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['black', 'black'], transform=proj,
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
axs.set_title(r'BMHW Cumulative Intensity 2008-2022 minus 1993-2007', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, max_diff_CAN, levels, cmap=cmap, transform=proj, extend ='both')
css_5=box_ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_cum_CAN[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'...\Fig_4\4f_BMHW_Cum_diff.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')
