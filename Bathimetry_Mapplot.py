# -*- coding: utf-8 -*-
"""

#########################      Bathymetry Mapplot      ########################

"""

#Loading required libraries
import numpy as np
import xarray as xr 

import cartopy.mpl.ticker as cticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
import cartopy.feature as cft

from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import matplotlib.colors as colors



## Load Bathymetry and MLD Data from Datos_ESMARES.py



############################
# Bottom depth / Elevation #
############################


## Total elevation ##

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)

cmap = cm.cm.deep

levels = np.linspace(0, 5500, num=23)

# Plot the main data
cs_1 = ax.contourf(LON_GC_bat, LAT_GC_bat, elevation_GC, levels, cmap=cmap, transform=proj, extend='max')
cs_2 = ax.contourf(LON_AL_bat, LAT_AL_bat, elevation_AL, levels, cmap=cmap, transform=proj, extend='max')
cs_3 = ax.contourf(LON_BAL_bat, LAT_BAL_bat, elevation_BAL, levels, cmap=cmap, transform=proj, extend='max')
cs_4 = ax.contourf(LON_NA_bat, LAT_NA_bat, elevation_NA, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['red', 'red'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_1 in el_1.collections:
    line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['red', 'red'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_2 in el_2.collections:
    line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['red', 'red'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_3 in el_3.collections:
    line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['red', 'red'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_4 in el_4.collections:
    line_4.set_zorder(2)  # Set the zorder of the contour lines to 2

# Plot the colorbar for main data
cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
cbar.set_label(r'Bottom depth [$m$]', fontsize=22)

# Add map features
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)

# Set the extent of the main plot
ax.set_extent([-16, 6.5, 34, 47], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
ax.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_bat, LAT_CAN_bat, elevation_CAN, levels, cmap=cmap, transform=proj, extend='max')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'red'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


#Save data so far
outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\Bathymetry_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



## Canarias ##

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(30)

cmap=cm.cm.deep

levels=np.linspace(0, 5500, num=23)

# Shapefile_Canarias.plot(ax=ax, facecolor='none', edgecolor='black')
cs= ax.contourf(LON_CAN_bat, LAT_CAN_bat, elevation_CAN, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


# cbar = plt.colorbar(cs, shrink=0.7, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
# cbar.set_label(r'Bottom depth [$m$]', fontsize=22)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
ax.set_extent([-22, -11, 24, 32.5], crs=proj)  #Canarias
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-20,-18,-16,-14,-12], crs=proj)
ax.set_yticks([24,26,28,30,32], crs=proj)
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('Bottom depth [$m$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\Bathymetry_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




## GC ##

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(30)

cmap=cm.cm.deep

levels=np.linspace(0, 5500, num=23)

cs= ax.contourf(LON_GC_bat, LAT_GC_bat, elevation_GC, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


# cbar = plt.colorbar(cs, shrink=0.7, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
# cbar.set_label(r'Bottom depth [$m$]', fontsize=22)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
ax.set_yticks([36, 37], crs=proj)
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('Bottom depth [$m$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\Bathymetry_GC.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





## AL ##

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=cm.cm.deep

levels=np.linspace(0, 5500, num=23)

# Shapefile_Canarias.plot(ax=ax, facecolor='none', edgecolor='black')
cs= ax.contourf(LON_AL_bat, LAT_AL_bat, elevation_AL, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


# cbar = plt.colorbar(cs, shrink=0.7, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
# cbar.set_label(r'Bottom depth [$m$]', fontsize=22)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-5, -4, -3, -2], crs=proj)
ax.set_yticks([35, 36], crs=proj) 
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('Bottom depth [$m$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\Bathymetry_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






## BAL ##

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(30)

cmap=cm.cm.deep

levels=np.linspace(0, 5500, num=23)

cs= ax.contourf(LON_BAL_bat, LAT_BAL_bat, elevation_BAL, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


# cbar = plt.colorbar(cs, shrink=0.7, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
# cbar.set_label(r'Bottom depth [$m$]', fontsize=22)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 38, 40, 42], crs=proj) 
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('Bottom depth [$m$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\Bathymetry_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








## NA ##

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=cm.cm.deep

levels=np.linspace(0, 5500, num=23)

cs= ax.contourf(LON_NA_bat, LAT_NA_bat, elevation_NA, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


# cbar = plt.colorbar(cs, shrink=0.5, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
# cbar.set_label(r'Bottom depth [$m$]', fontsize=22)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
ax.set_extent([-14, -1.5, 41, 47], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-12, -10, -8, -6, -4, -2], crs=proj) 
ax.set_yticks([41, 43, 45, 47], crs=proj)  
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('Bottom depth [$m$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\Bathymetry_NA.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








#########
## MLD ##
#########

# Aver_MLD_GC = np.nanmean(MLD_GC, axis=0)
# Aver_MLD_AL = np.nanmean(MLD_AL, axis=0)
# Aver_MLD_BAL = np.nanmean(MLD_BAL, axis=0)
# Aver_MLD_NA = np.nanmean(MLD_NA, axis=0)
# Aver_MLD_CAN = np.nanmean(MLD_CAN, axis=0)


Max_MLD_GC = MLD_GC.max(dim='time', skipna=True)
Max_MLD_AL = MLD_AL.max(dim='time', skipna=True)
Max_MLD_BAL = MLD_BAL.max(dim='time', skipna=True)
Max_MLD_NA = MLD_NA.max(dim='time', skipna=True)
Max_MLD_CAN = MLD_CAN.max(dim='time', skipna=True)



## Total MLD ##

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(18)

cmap = cm.cm.dense

levels = np.linspace(0, 550, num=23)

# Plot the main data
cs_1 = ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, Max_MLD_GC, levels, cmap=cmap, transform=proj, extend='max')
cs_2 = ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, Max_MLD_AL, levels, cmap=cmap, transform=proj, extend='max')
cs_3 = ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, Max_MLD_BAL, levels, cmap=cmap, transform=proj, extend='max')
cs_4 = ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, Max_MLD_NA, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['red', 'red'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_1 in el_1.collections:
    line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['red', 'red'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_2 in el_2.collections:
    line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['red', 'red'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_3 in el_3.collections:
    line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['red', 'red'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_4 in el_4.collections:
    line_4.set_zorder(2)  # Set the zorder of the contour lines to 2

# Plot the colorbar for main data
cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 100, 200, 300, 400, 500])
cbar.set_label(r'Density ML thickness [$m$]', fontsize=22)

# Add map features
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)

# Set the extent of the main plot
ax.set_extent([-16, 6.5, 34, 47], crs=proj)
lon_formatter = cticker.LongitudeFormatter(zero_direction_label=True)
lat_formatter = cticker.LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
ax.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.078, 0.145, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Max_MLD_CAN, levels, cmap=cmap, transform=proj, extend='max')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'red'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
# box_ax.add_feature(cft.BORDERS)


# Save the figure so far
outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\MLD_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







## Canarias MLD ##

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=cm.cm.dense

levels=np.linspace(0, 550, num=23)

# Shapefile_Canarias.plot(ax=ax, facecolor='none', edgecolor='black')
cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Max_MLD_CAN, levels, cmap=cmap, transform=proj, extend ='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1.25)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs, shrink=1, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 100, 200, 300, 400, 500])
cbar.set_label(r'Density ML thickness [$m$]', fontsize=22)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
ax.set_yticks([24,26,28,30,32], crs=proj)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\MLD_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




## South Atlantic ##

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=cm.cm.deep

levels=np.linspace(0, 550, num=23)

# Shapefile_Canarias.plot(ax=ax, facecolor='none', edgecolor='black')
cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Max_MLD_CAN, levels, cmap=cmap, transform=proj, extend ='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1.25)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs, shrink=1, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 100, 200, 300, 400, 500])
cbar.set_label(r'Density ML thickness [$m$]', fontsize=22)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
ax.set_yticks([24,26,28,30,32], crs=proj)
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('Bottom depth [$m$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\MLD_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










                        ### Subplots ###

## Bottom depth ##

# Create a figure and set the size (adjust as needed)
fig = plt.figure(figsize=(10, 24))

# Define the relative heights and widths of the rows and columns
row_heights = [2, 1]
col_widths = [2, 2, 2]  # Equal widths for all columns

# Create GridSpec with the specified layout and adjustments
gs = GridSpec(2, 3, height_ratios=row_heights, width_ratios=col_widths, figure=fig, wspace=0.3, hspace=0.3)

proj = ccrs.PlateCarree()  # Choose the projection

# Create subplots using the GridSpec and add your own content to each subplot
ax1 = plt.subplot(gs[0, 0], projection=proj)

ax2 = plt.subplot(gs[0, 1], projection=proj)

ax3 = plt.subplot(gs[1, 0], projection=proj)

ax4 = plt.subplot(gs[1, 1], projection=proj)

ax5 = plt.subplot(gs[1, 2], projection=proj)

# Fine-tune the figure appearance for publication in Nature
plt.subplots_adjust(left=0.08, right=0.92, bottom=0.6, top=0.75, wspace=0.3, hspace=0.3)
plt.rc('font', family='Arial', size=10)  # Set font and font size for all text elements
plt.rc('axes', labelsize=10)  # Set font size for axis labels
plt.rc('xtick', labelsize=10)  # Set font size for x-axis tick labels
plt.rc('ytick', labelsize=10)  # Set font size for y-axis tick labels

# Ajustamos la alineación de los subplots en su fila
# plt.tight_layout()

# Now you can plot your data on each subplot as needed using the ax objects.
# Plot data on ax1:
cmap=cm.cm.deep
levels=np.linspace(0, 5500, num=23)

#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)

cs1= ax1.contourf(LON_NA_bat, LAT_NA_bat, elevation_NA, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax1.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

ax1.coastlines(resolution='10m', color='black', linewidth=1)
ax1.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
ax1.set_extent([-14, -1.5, 41, 47], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
ax1.set_xticks([-12, -10, -8, -6, -4, -2], crs=proj) 
ax1.set_yticks([41, 43, 45, 47], crs=proj)  

# Plot data on ax2:
cs2= ax2.contourf(LON_AL_bat, LAT_AL_bat, elevation_AL, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax2.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

ax2.coastlines(resolution='10m', color='black', linewidth=1)
ax2.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
ax2.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
ax2.set_xticks([ -5, -4, -3, -2], crs=proj)
ax2.set_yticks([35, 36], crs=proj) 

# Plot data on ax3:
cs3= ax3.contourf(LON_CAN_bat, LAT_CAN_bat, elevation_CAN, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax3.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


ax3.coastlines(resolution='10m', color='black', linewidth=1)
ax3.add_feature(land_10m)
ax3.set_extent([-22, -11, 24, 32.5], crs=proj)  #Canarias
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax3.xaxis.set_major_formatter(lon_formatter)
ax3.yaxis.set_major_formatter(lat_formatter)
ax3.set_xticks([-20,-18,-16,-14,-12], crs=proj)
ax3.set_yticks([24, 26, 28, 30, 32], crs=proj)

# Plot data on ax4:
cs4= ax4.contourf(LON_GC_bat, LAT_GC_bat, elevation_GC, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax4.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

ax4.coastlines(resolution='10m', color='black', linewidth=1)
ax4.add_feature(land_10m)
ax4.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax4.xaxis.set_major_formatter(lon_formatter)
ax4.yaxis.set_major_formatter(lat_formatter)
ax4.set_xticks([-7.5, -6.5, -5.5], crs=proj)
ax4.set_yticks([36, 37], crs=proj)

# Plot data on ax5:
cs5= ax5.contourf(LON_BAL_bat, LAT_BAL_bat, elevation_BAL, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax5.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


ax5.coastlines(resolution='10m', color='black', linewidth=1)
ax5.add_feature(land_10m)
ax5.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax5.xaxis.set_major_formatter(lon_formatter)
ax5.yaxis.set_major_formatter(lat_formatter)
ax5.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax5.set_yticks([36, 38, 40, 42], crs=proj) 


# Optionally, you can add colorbars, legends, and other annotations as needed
levels=np.linspace(0, 5500, num=23)
norm = colors.BoundaryNorm(levels, ncolors=cm.cm.deep.N)
cmap = plt.cm.ScalarMappable(norm=norm, cmap=cm.cm.deep)

# Create common colorbar
cbar = plt.colorbar(cmap, ax=[ax1, ax2, ax3, ax4, ax5], pad=0.02, aspect=22, extend ='max', format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=6, direction='in', labelsize=12)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
cbar.set_label(r'Bottom depth [$m$]', fontsize=12)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Paper_EEMM\Fig_1\Bathymetry_subplots.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






































######################
## MLD/Bottom Depth ##
######################

#Interpolate Bathymetry ds grid to the GLORYS dataset grid
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



MLD_Bottom_GC = Max_MLD_GC/elevation_GC_interp
MLD_Bottom_AL = Max_MLD_AL/elevation_AL_interp
MLD_Bottom_BAL = Max_MLD_BAL/elevation_BAL_interp
MLD_Bottom_NA = Max_MLD_NA/elevation_NA_interp
MLD_Bottom_CAN = Max_MLD_CAN/elevation_CAN_interp




fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=cm.cm.deep_r
# cmap=plt.cm.YlGnBu_r
# cmap=plt.cm.gist_earth
# cmap=cm.cm.haline
# cmap=cm.cm.topo




levels = np.linspace(0, 1, num=21)


cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MLD_Bottom_GC, levels, cmap=cmap, transform=proj, extend ='max')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MLD_Bottom_AL, levels, cmap=cmap, transform=proj, extend ='max')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MLD_Bottom_BAL, levels, cmap=cmap, transform=proj, extend ='max')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MLD_Bottom_NA, levels, cmap=cmap, transform=proj, extend ='max')


# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=12, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_2, inline=True, fontsize=12, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_3, inline=True, fontsize=12, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_4, inline=True, fontsize=12, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2



cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22) 
cbar.ax.minorticks_off()

cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_label(r'Density ML thickness / Bottom depth', fontsize=22)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-16, 6.5, 34, 47], crs=proj) 
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj) 
ax.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj) 
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title(r'Density ML thickness [$m$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_Bathymetry\Figuras_Bathymetry\MLD_Bathy_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






## Canarias ##

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=cm.cm.deep_r

levels=np.linspace(0, 1, num=21)

# Shapefile_Canarias.plot(ax=ax, facecolor='none', edgecolor='black')
cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MLD_Bottom_CAN, levels, cmap=cmap, transform=proj, extend ='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'red'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs, shrink=1, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_label(r'Density ML thickness / Bottom depth', fontsize=22)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
ax.set_yticks([24,26,28,30,32], crs=proj)
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('Bottom depth [$m$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_Bathymetry\Figuras_Bathymetry\MLD_Bathy_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



















