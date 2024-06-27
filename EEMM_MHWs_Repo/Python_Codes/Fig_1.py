# -*- coding: utf-8 -*-
"""

#########################      Figure 1 in 
Fernández-Barba, M., Huertas, I. E., & Navarro, G. (2024). 
Assessment of surface and bottom marine heatwaves along the Spanish coast. 
Ocean Modelling, 190, 102399.                          ########################

"""

#Loading required libraries
import numpy as np

import cartopy.mpl.ticker as cticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
import cartopy.feature as cft

import matplotlib.ticker as ticker



## Load Bathymetry and MLT Data from Datos_ESMARES.py




#############
## Max MLT ##  (Figure 1a)
#############

#Calculating Maximum MLT
Max_MLT_SA = MLT_SA.max(dim='time', skipna=True)
Max_MLT_AL = MLT_AL.max(dim='time', skipna=True)
Max_MLT_BAL = MLT_BAL.max(dim='time', skipna=True)
Max_MLT_NA = MLT_NA.max(dim='time', skipna=True)
Max_MLT_CAN = MLT_CAN.max(dim='time', skipna=True)



## Total MLT ##

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
cs_1 = ax.contourf(LON_SA_MODEL, LAT_SA_MODEL, Max_MLT_SA, levels, cmap=cmap, transform=proj, extend='max')
cs_2 = ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, Max_MLT_AL, levels, cmap=cmap, transform=proj, extend='max')
cs_3 = ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, Max_MLT_BAL, levels, cmap=cmap, transform=proj, extend='max')
cs_4 = ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, Max_MLT_NA, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['red', 'blue'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_1 in el_1.collections:
    line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['red', 'blue'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_2 in el_2.collections:
    line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['red', 'blue'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_3 in el_3.collections:
    line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['red', 'blue'], transform=proj,
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
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Max_MLT_CAN, levels, cmap=cmap, transform=proj, extend='max')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'blue'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
# box_ax.add_feature(cft.BORDERS)


# Save the figure so far
outfile = r'...\Fig_1\MLT_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')





############################
# Bottom depth / Elevation #  (Figure 1b-f)
############################

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

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'blue'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1.75)
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


outfile = r'...\Fig_1\Bathymetry_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




## Golfo de Cadiz ##

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

cs= ax.contourf(LON_SA_bat, LAT_SA_bat, elevation_SA, levels, cmap=cmap, transform=proj, extend='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = ax.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['red', 'blue'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1.75)
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


outfile = r'...\Fig_1\Bathymetry_SA.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')





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

el_1 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['red', 'blue'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1.75)
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


outfile = r'...\Fig_1\Bathymetry_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')






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

el_1 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['red', 'blue'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1.75)
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


outfile = r'...\Fig_1\Bathymetry_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')








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

el_1 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['red', 'blue'], transform=proj, linestyles=['solid', 'dashed'], linewidths=1.75)
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


outfile = r'...\Fig_1\Bathymetry_NA.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')
