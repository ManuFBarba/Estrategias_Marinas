# -*- coding: utf-8 -*-
"""

########################## Fig. 1. Study Area ##########################

"""

#Loading required Python modules
import numpy as np
import xarray as xr 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LogNorm

import cmocean as cm

import cartopy.crs as ccrs
import cartopy.feature as cft


## Loading lat and lon arrays and Bathymetric data from 'CHL_Phenology_metrics.py'

## Loading previously-processed MLD and CHL datasets
Max_MLD_CAN = np.load(r'E:\...\MLD_Data\Max_MLD/Max_MLD_CAN.npy')
Max_MLD_NA = np.load(r'E:\...\MLD_Data\Max_MLD/Max_MLD_NA.npy')
Max_MLD_SA = np.load(r'E:\...\MLD_Data\Max_MLD/Max_MLD_SA.npy')
Max_MLD_AL = np.load(r'E:\...\MLD_Data\Max_MLD/Max_MLD_AL.npy')
Max_MLD_BAL = np.load(r'E:\...\MLD_Data\Max_MLD/Max_MLD_BAL.npy')

ds_CHL_CAN = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_CAN.nc')
ds_CHL_NA = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_NA.nc')
ds_CHL_SA = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_SA.nc')
ds_CHL_AL = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_AL.nc')
ds_CHL_BAL = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_BAL.nc')
# Mean_CHL_CAN = ds_CHL_CAN.CHL.mean(dim='time', skipna=True).to_netcdf(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_CAN.nc')
# Mean_CHL_NA = ds_CHL_NA.CHL.mean(dim='time', skipna=True).to_netcdf(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_NA.nc')
# Mean_CHL_SA = ds_CHL_SA.CHL.mean(dim='time', skipna=True).to_netcdf(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_SA.nc')
# Mean_CHL_AL = ds_CHL_AL.CHL.mean(dim='time', skipna=True).to_netcdf(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_AL.nc')
# Mean_CHL_BAL = ds_CHL_BAL.CHL.mean(dim='time', skipna=True).to_netcdf(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_BAL.nc')

Mean_CHL_CAN = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_CAN.nc')
Mean_CHL_NA = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_NA.nc')
Mean_CHL_SA = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_SA.nc')
Mean_CHL_AL = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_AL.nc')
Mean_CHL_BAL = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/Mean_CHL_BAL.nc')



## Fig. 1a (Maximum Mixed Layer Depth)
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=cm.cm.dense
levels = np.arange(0, 350, 10)

cs_1= axs.contourf(LON_SA, LAT_SA, np.nanmean(Max_MLD_SA, axis=2), cmap=cmap, levels=levels, transform=proj, extend ='max')
cs_2= axs.contourf(LON_AL, LAT_AL, np.nanmean(Max_MLD_AL, axis=2), cmap=cmap, levels=levels, transform=proj, extend ='max')
cs_3= axs.contourf(LON_BAL, LAT_BAL, np.nanmean(Max_MLD_BAL, axis=2), cmap=cmap, levels=levels, transform=proj, extend ='max')
cs_4= axs.contourf(LON_NA, LAT_NA, np.nanmean(Max_MLD_NA, axis=2), cmap=cmap, levels=levels, transform=proj, extend ='max')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.0f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=20)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 50, 100, 150, 200, 250, 300])

cbar.set_label(r'MLD [m]', fontsize=20)
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

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the           [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, np.nanmean(Max_MLD_CAN, axis=2), levels=levels, cmap=cmap, transform=proj, extend ='max')

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_1\Fig_1a.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Testing Mean CHL (colorbar)
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.Spectral_r
norm=LogNorm(vmin=0.1, vmax=2.5)
levels = np.logspace(np.log10(0.1), np.log10(2.5), 65)

cs_1= axs.contourf(LON_NA, LAT_NA, Mean_CHL_NA.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')
cs_2= axs.contourf(LON_SA, LAT_SA, Mean_CHL_SA.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')
cs_3= axs.contourf(LON_AL, LAT_AL, Mean_CHL_AL.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')
cs_4= axs.contourf(LON_BAL, LAT_BAL, Mean_CHL_BAL.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2, 4], zorder=2)
# axs.clabel(el_1, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2, 4], zorder=2)
# axs.clabel(el_2, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2, 4], zorder=2)
# axs.clabel(el_3, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2, 4], zorder=2)
# axs.clabel(el_4, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')


cbar = plt.colorbar(cs_1, shrink=0.90, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=20)
cbar.ax.minorticks_off()
cbar.set_ticks([0.1, 0.2, 0.5, 1, 2.5])

cbar.set_label(r'[Chl-a] [mg·m$^{-3}$]', fontsize=20)
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

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the           [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN, LAT_CAN, Mean_CHL_CAN.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el_5, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_1\cb_CHL.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')




## Fig. 1b: Mean Chl-a (NA)
fig = plt.figure(figsize=(5, 5))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(13)

cmap=plt.cm.Spectral_r
norm=LogNorm(vmin=0.15, vmax=2.5)
levels = np.logspace(np.log10(0.15), np.log10(2.5), 65)

cs= ax.contourf(LON_NA, LAT_NA, Mean_CHL_NA.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[1.5], zorder=2)
# axs.clabel(el, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# cbar = plt.colorbar(cs, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
# cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
# cbar.ax.minorticks_off()
# cbar.set_ticks([0.2, 0.5, 1, 2, 5])
# cbar.set_ticklabels(['0.2', '0.5', '1.0', '2.0', '5.0'])
# cbar.set_label(r'$[mg \cdot m^{-3}]$', fontsize=40)

# Add map features
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)

ax.set_extent([-14, -1.5, 41, 47], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-12, -8, -4], crs=proj) 
ax.set_yticks([41, 43, 45, 47], crs=proj)  
#Set the title
plt.title('North Atlantic (NA)', fontsize=25)
plt.show()

outfile = r'E:\...\Figures\Fig_1\Fig_1b.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 1c: Mean Chl-a (AL)
fig = plt.figure(figsize=(5, 5))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(12.5)

cmap=plt.cm.Spectral_r
norm=LogNorm(vmin=0.15, vmax=2.5)
levels = np.logspace(np.log10(0.15), np.log10(2.5), 65)

cs= ax.contourf(LON_AL, LAT_AL, Mean_CHL_AL.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[1.5], zorder=2)
# axs.clabel(el, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# cbar = plt.colorbar(cs, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), 
#                     pad=0.02, extend='both', aspect=10)
# cbar.ax.tick_params(axis='y', size=7, direction='in', labelsize=40)
# cbar.ax.minorticks_off()
# cbar.set_ticks([0.2, 0.5, 1, 2, 5])
# cbar.set_ticklabels(['0.2', '0.5', '1.0', '2.0', '5.0'])
# cbar.set_label(r'[mg $\cdot$ m$^{-3}$]', fontsize=40)

# Add map features
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)

ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-5, -4, -3, -2], crs=proj)
ax.set_yticks([35, 36], crs=proj) 
#Set the title
plt.title('SoG and Alboran Sea (AL)', fontsize=25)
plt.show()

outfile = r'E:\...\Figures\Fig_1\Fig_1c.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')




## Fig. 1d: Mean Chl-a (CAN)
fig = plt.figure(figsize=(5, 5))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(22)

cmap=plt.cm.Spectral_r
norm=LogNorm(vmin=0.1, vmax=2.5)
levels = np.logspace(np.log10(0.1), np.log10(2.5), 65)

cs= ax.contourf(LON_CAN, LAT_CAN, Mean_CHL_CAN.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[1.5], zorder=2)
# axs.clabel(el, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# cbar = plt.colorbar(cs, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
# cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
# cbar.ax.minorticks_off()
# cbar.set_ticks([0.2, 0.5, 1, 2, 5])
# cbar.set_ticklabels(['0.2', '0.5', '1.0', '2.0', '5.0'])
# cbar.set_label(r'$[mg \cdot m^{-3}]$', fontsize=40)

# Add map features
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)

ax.set_extent([-22, -11, 24, 32.5], crs=proj)  #Canarias
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-20,-16,-12], crs=proj)
ax.set_yticks([24,26,28,30,32], crs=proj)
#Set the title
plt.title('Canary (CAN)', fontsize=25)
plt.show()

outfile = r'E:\...\Figures\Fig_1\Fig_1d.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 1e: Mean Chl-a (SA)
fig = plt.figure(figsize=(5, 5))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(22)

cmap=plt.cm.Spectral_r
norm=LogNorm(vmin=0.15, vmax=2.5)
levels = np.logspace(np.log10(0.15), np.log10(2.5), 65)

cs= ax.contourf(LON_SA, LAT_SA, Mean_CHL_SA.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el = ax.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
axs.clabel(el, fmt='%d', fontsize=20, inline=True, inline_spacing=30, colors='crimson')

# cbar = plt.colorbar(cs, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
# cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
# cbar.ax.minorticks_off()
# cbar.set_ticks([0.2, 0.5, 1, 2, 5])
# cbar.set_ticklabels(['0.2', '0.5', '1.0', '2.0', '5.0'])
# cbar.set_label(r'$[mg \cdot m^{-3}]$', fontsize=40)

# Add map features
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)

ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
ax.set_yticks([36, 37], crs=proj) 
#Set the title
plt.title('South Atlantic (SA)', fontsize=25)
plt.show()

outfile = r'E:\...\Figures\Fig_1\Fig_1e.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')


## Fig. 1f: Mean Chl-a (BAL)
fig = plt.figure(figsize=(5, 5))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(22)

cmap=plt.cm.Spectral_r
norm=LogNorm(vmin=0.15, vmax=2.5)
levels = np.logspace(np.log10(0.15), np.log10(2.5), 65)

cs= ax.contourf(LON_BAL, LAT_BAL, Mean_CHL_BAL.CHL, cmap=cmap, norm=norm, levels=levels, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [200]

el = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['crimson'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)
# axs.clabel(el, fmt='%d', fontsize=30, inline=True, inline_spacing=-20, colors='black')

# cbar = plt.colorbar(cs, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
# cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
# cbar.ax.minorticks_off()
# cbar.set_ticks([0.2, 0.5, 1, 2, 5])
# cbar.set_ticklabels(['0.2', '0.5', '1.0', '2.0', '5.0'])
# cbar.set_label(r'$[mg \cdot m^{-3}]$', fontsize=40)

# Add map features
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)

ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 38, 40, 42], crs=proj) 
#Set the title
plt.title('Levantine-Balearic (BAL)', fontsize=25)
plt.show()

outfile = r'E:\...\Figures\Fig_1\Fig_1f.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')

