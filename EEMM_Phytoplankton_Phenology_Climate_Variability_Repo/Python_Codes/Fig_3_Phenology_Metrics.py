# -*- coding: utf-8 -*-
"""

############### Fig. 3. Phytoplankton Bloom Phenology Metrics #################

"""

#Loading required Python modules
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import BoundaryNorm

import cmocean as cm

import cartopy.crs as ccrs
import cartopy.feature as cft

import seaborn as sns


## Load 'CHL_Phenology_metrics.py'

## Fig. 3a (Bloom Max Chl-a)
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=plt.cm.Spectral_r
norm=LogNorm(vmin=0.2, vmax=5)

cs_1= axs.pcolormesh(LON_NA, LAT_NA, np.nanmean(bloom_max_NA, axis=2), norm=norm, cmap=cmap, transform=proj)
cs_2= axs.pcolormesh(LON_SA, LAT_SA, np.nanmean(bloom_max_SA, axis=2), norm=norm, cmap=cmap, transform=proj)
cs_3= axs.pcolormesh(LON_AL, LAT_AL, np.nanmean(bloom_max_AL, axis=2), norm=norm, cmap=cmap, transform=proj)
cs_4= axs.pcolormesh(LON_BAL, LAT_BAL, np.nanmean(bloom_max_BAL, axis=2), norm=norm, cmap=cmap, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0.2, 0.5, 1, 2, 5])

cbar.set_label(r'$\mathrm{[mg \cdot m^{-3}]}$', fontsize=40)
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
axs.set_title('BMaxChl-a', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the           [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, np.nanmean(bloom_max_CAN, axis=2), norm=norm, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3a.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3b (Bloom Max Chl-a 2011-2023 minus 1998-2010)
max_diff_SA = np.nanmean(bloom_max_SA[:,:,13:26], axis=2) - np.nanmean(bloom_max_SA[:,:,0:13], axis=2)
max_diff_AL = np.nanmean(bloom_max_AL[:,:,13:26], axis=2) - np.nanmean(bloom_max_AL[:,:,0:13], axis=2)
max_diff_BAL = np.nanmean(bloom_max_BAL[:,:,13:26], axis=2) - np.nanmean(bloom_max_BAL[:,:,0:13], axis=2)
max_diff_NA = np.nanmean(bloom_max_NA[:,:,13:26], axis=2) - np.nanmean(bloom_max_NA[:,:,0:13], axis=2)
max_diff_CAN = np.nanmean(bloom_max_CAN[:,:,13:26], axis=2) - np.nanmean(bloom_max_CAN[:,:,0:13], axis=2)

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.delta
norm = TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75)

cs_1= axs.pcolormesh(LON_NA, LAT_NA, max_diff_NA, cmap=cmap, norm=norm, transform=proj)
cs_2= axs.pcolormesh(LON_SA, LAT_SA, max_diff_SA, cmap=cmap, norm=norm, transform=proj)
cs_3= axs.pcolormesh(LON_AL, LAT_AL, max_diff_AL, cmap=cmap, norm=norm, transform=proj)
cs_4= axs.pcolormesh(LON_BAL, LAT_BAL, max_diff_BAL, cmap=cmap, norm=norm, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.2f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])

cbar.set_label(r'$\mathrm{[mg \cdot m^{-3}]}$', fontsize=40)
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
axs.set_title('BMaxChl-a 2011-2023 minus 1998-2010', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the           [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, max_diff_CAN, cmap=cmap, norm=norm, transform=proj)
# css_5=box_ax.scatter(LON_CAN[::4,::4], LAT_CAN[::4,::4], signif_max_CAN[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3b.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3c (Density Distributions Bloom Max Chl-a)
years = np.arange(1998, 2024)

fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
data_demarcations = [bloom_max_NA, bloom_max_AL, bloom_max_CAN, bloom_max_SA, bloom_max_BAL]
labels = ['North Atlantic (NA)', 'SoG and Alboran Sea (AL)', 'Canary (CAN)', 'South Atlantic (SA)',  'Levantine-Balearic (BAL)']

cmap = plt.cm.Spectral_r
colors = cmap(np.linspace(0, 1, len(years)))

for i, data in enumerate(data_demarcations):
    for j, year in enumerate(years):
        data_flat = data[:, :, j].ravel()  
        
        data_flat = data_flat[~np.isnan(data_flat)]
        
        if len(data_flat) > 0:
            sns.kdeplot(data_flat, color=colors[j], linewidth=2, ax=axs[i])
    
    axs[i].set_xscale('log')  
    axs[i].set_xlim(0.1, 10)  
    axs[i].set_title(labels[i], fontsize=45)  
    axs[i].set_xlabel(r'BMaxChl-a $\mathrm{[mg \cdot m^{-3}]}$', fontsize=45)
    axs[i].set_ylabel('')
    

    axs[i].set_xticks([0.25, 1, 5, 10])
    axs[i].set_xticklabels([0.25, 1, 5, 10], fontsize=45)  
    axs[i].set_yticks([0, 2.5])
    axs[i].set_yticklabels([0, 2.5], fontsize=45)  
    
    axs[i].tick_params(axis='both', which='major', labelsize=45)  

    if i == 2:  
        axs[i].set_ylabel('Kernel Density', fontsize=45, labelpad=39)  

        axs[i].set_yticks([0, 2.5, 5])
        axs[i].set_yticklabels([0, '', 5], fontsize=45) 

    if i == 4:  
        axs[i].set_yticks([0, 2.5, 5])
        axs[i].set_yticklabels([0, '',5], fontsize=45) 
        
# Colorbar customization
cbar_ax = fig.add_axes([0.91, 0.11, 0.025, 0.77])  
norm = BoundaryNorm(boundaries=np.arange(1998, 2025, 1), ncolors=cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
major_ticks = np.arange(1998, 2025, 4)
cbar.set_ticks(major_ticks)  
cbar.set_ticklabels(major_ticks)
cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
cbar.ax.tick_params(labelsize=45, direction='in', which='major', length=13, width=2)

plt.subplots_adjust(hspace=0.6)
plt.show()


outfile = r'E:\...\Figures\Fig_3\Fig_3c.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3d (Bloom Initiation)
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

# cmap=plt.cm.Paired
cmap=plt.cm.terrain
vmin, vmax = 0, 244

cs_1= axs.pcolormesh(LON_SA, LAT_SA, np.nanmean(bloom_ini_SA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, np.nanmean(bloom_ini_AL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, np.nanmean(bloom_ini_BAL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, np.nanmean(bloom_ini_NA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 30.5, 59.0, 91.5, 122.0, 152.5, 183.0, 213.5, 244.0])
cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])

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
axs.set_title('BInit', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, np.nanmean(bloom_ini_CAN, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3d.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3e (Bloom Initiation 2011-2023 minus 1998-2010)
ini_diff_SA = np.nanmean(bloom_ini_SA[:,:,13:26], axis=2) - np.nanmean(bloom_ini_SA[:,:,0:13], axis=2)
ini_diff_AL = np.nanmean(bloom_ini_AL[:,:,13:26], axis=2) - np.nanmean(bloom_ini_AL[:,:,0:13], axis=2)
ini_diff_BAL = np.nanmean(bloom_ini_BAL[:,:,13:26], axis=2) - np.nanmean(bloom_ini_BAL[:,:,0:13], axis=2)
ini_diff_NA = np.nanmean(bloom_ini_NA[:,:,13:26], axis=2) - np.nanmean(bloom_ini_NA[:,:,0:13], axis=2)
ini_diff_CAN = np.nanmean(bloom_ini_CAN[:,:,13:26], axis=2) - np.nanmean(bloom_ini_CAN[:,:,0:13], axis=2)


fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.delta
norm = TwoSlopeNorm(vmin=-90, vcenter=0, vmax=90)

cs_1= axs.pcolormesh(LON_SA, LAT_SA, ini_diff_SA, cmap=cmap, norm=norm, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, ini_diff_AL, cmap=cmap, norm=norm, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, ini_diff_BAL, cmap=cmap, norm=norm, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, ini_diff_NA, cmap=cmap, norm=norm, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.98, format=ticker.FormatStrFormatter('%.0f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([-90, -60, -30, 0, 30, 60, 90])
cbar.set_label(r'[days]', fontsize=40)
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
axs.set_title('BInit 2011-2023 minus 1998-2010', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, ini_diff_CAN, cmap=cmap, norm=norm, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3e.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3f (Density Distributions Bloom Initiation)
years = np.arange(1998, 2024)

fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
data_demarcations = [bloom_ini_NA, bloom_ini_AL, bloom_ini_CAN, bloom_ini_SA, bloom_ini_BAL]
labels = ['North Atlantic (NA)', 'SoG and Alboran Sea (AL)', 'Canary (CAN)', 'South Atlantic (SA)',  'Levantine-Balearic (BAL)']

cmap = plt.cm.Spectral_r
colors = cmap(np.linspace(0, 1, len(years)))

for i, data in enumerate(data_demarcations):
    for j, year in enumerate(years):
        data_flat = data[:, :, j].ravel()  
        
        data_flat = data_flat[~np.isnan(data_flat)]
        
        if len(data_flat) > 0:
            sns.kdeplot(data_flat, color=colors[j], linewidth=2, ax=axs[i])
    
    axs[i].set_xlim(0, 366)  
    axs[i].set_title(labels[i], fontsize=45)  
    axs[i].set_xlabel(r'BInit $\mathrm{[days]}$', fontsize=45)
    axs[i].set_ylabel('')
    
    axs[i].set_xticks([-30, 0, 30.5, 59.0, 91.5, 122.0, 152.5, 183.0, 213.5, 244.0, 274.5, 305.0, 336])
    axs[i].set_xticklabels(['', 'Jan', '', 'Mar', '', 'May', '', 'Jul', '', 'Sep', '', '', 'Dec'], fontsize=45)   
    
    axs[i].tick_params(axis='both', which='major', labelsize=45)  

    if i == 0:  
        axs[i].set_yticklabels([0, 0.1], fontsize=45) 

    if i == 1:  
        axs[i].set_yticklabels([0, 0.03], fontsize=45) 

    if i == 2:  
        axs[i].set_ylabel('Kernel Density', fontsize=45, labelpad=30)  
        axs[i].set_yticklabels([0, 0.5], fontsize=45) 

    if i == 3:  
        axs[i].set_yticklabels([0, 0.05], fontsize=45) 

    if i == 4:  
        axs[i].set_yticklabels([0, 0.25], fontsize=45) 
        
# Colorbar customization
cbar_ax = fig.add_axes([0.91, 0.11, 0.025, 0.77])  
norm = BoundaryNorm(boundaries=np.arange(1998, 2025, 1), ncolors=cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
major_ticks = np.arange(1998, 2025, 4)
cbar.set_ticks(major_ticks)  
cbar.set_ticklabels(major_ticks)
cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
cbar.ax.tick_params(labelsize=45, direction='in', which='major', length=13, width=2)

plt.subplots_adjust(hspace=0.6)
plt.show()


outfile = r'E:\...\Figures\Fig_3\Fig_3f.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')




## Fig. 3g (Bloom Termination)
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

# cmap=plt.cm.Paired
cmap=plt.cm.terrain
vmin, vmax = 91.5, 274.5

cs_1= axs.pcolormesh(LON_SA, LAT_SA, np.nanmean(bloom_fin_SA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, np.nanmean(bloom_fin_AL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, np.nanmean(bloom_fin_BAL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, np.nanmean(bloom_fin_NA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([91.5, 122.0, 152.5, 183.0, 213.5, 244.0, 274.5])
cbar.set_ticklabels(['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Nov'])

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
axs.set_title('BTerm', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, np.nanmean(bloom_fin_CAN, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3g.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3h (Bloom Termination 2011-2023 minus 1998-2010)
fin_diff_SA = np.nanmean(bloom_fin_SA[:,:,13:26], axis=2) - np.nanmean(bloom_fin_SA[:,:,0:13], axis=2)
fin_diff_AL = np.nanmean(bloom_fin_AL[:,:,13:26], axis=2) - np.nanmean(bloom_fin_AL[:,:,0:13], axis=2)
fin_diff_BAL = np.nanmean(bloom_fin_BAL[:,:,13:26], axis=2) - np.nanmean(bloom_fin_BAL[:,:,0:13], axis=2)
fin_diff_NA = np.nanmean(bloom_fin_NA[:,:,13:26], axis=2) - np.nanmean(bloom_fin_NA[:,:,0:13], axis=2)
fin_diff_CAN = np.nanmean(bloom_fin_CAN[:,:,13:26], axis=2) - np.nanmean(bloom_fin_CAN[:,:,0:13], axis=2)


fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.delta
norm = TwoSlopeNorm(vmin=-90, vcenter=0, vmax=90)

cs_1= axs.pcolormesh(LON_SA, LAT_SA, fin_diff_SA, cmap=cmap, norm=norm, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, fin_diff_AL, cmap=cmap, norm=norm, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, fin_diff_BAL, cmap=cmap, norm=norm, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, fin_diff_NA, cmap=cmap, norm=norm, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.98, format=ticker.FormatStrFormatter('%.0f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([-90, -60, -30, 0, 30, 60, 90])
cbar.set_label(r'[days]', fontsize=40)
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
axs.set_title('BTerm 2011-2023 minus 1998-2010', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, fin_diff_CAN, cmap=cmap, norm=norm, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3h.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3i (Density Distributions Bloom Termination)
years = np.arange(1998, 2024)

fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
data_demarcations = [bloom_fin_NA, bloom_fin_AL, bloom_fin_CAN, bloom_fin_SA, bloom_fin_BAL]
labels = ['North Atlantic (NA)', 'SoG and Alboran Sea (AL)', 'Canary (CAN)', 'South Atlantic (SA)',  'Levantine-Balearic (BAL)']

cmap = plt.cm.Spectral_r
colors = cmap(np.linspace(0, 1, len(years)))

for i, data in enumerate(data_demarcations):
    for j, year in enumerate(years):
        data_flat = data[:, :, j].ravel()  
        
        data_flat = data_flat[~np.isnan(data_flat)]
        
        if len(data_flat) > 0:
            sns.kdeplot(data_flat, color=colors[j], linewidth=2, ax=axs[i])
    
    axs[i].set_xlim(0, 366)  
    axs[i].set_title(labels[i], fontsize=45)  
    axs[i].set_xlabel(r'BTerm $\mathrm{[days]}$', fontsize=45)
    axs[i].set_ylabel('')
    
    axs[i].set_xticks([0, 30.5, 59.0, 91.5, 122.0, 152.5, 183.0, 213.5, 244.0, 274.5, 305.0, 336])
    axs[i].set_xticklabels(['', 'Feb', '', 'Apr', '', 'Jun', '', 'Aug', '', 'Oct', '', 'Dec'], fontsize=45)   
    
    axs[i].tick_params(axis='both', which='major', labelsize=45)  

    if i == 0:  
        axs[i].set_yticklabels([0, 0.1], fontsize=45) 

    if i == 1:  
        axs[i].set_yticklabels([0, 0.03], fontsize=45) 

    if i == 2:  
        axs[i].set_ylabel('Kernel Density', fontsize=45, labelpad=10)  
        axs[i].set_yticklabels([0, 0.05], fontsize=45) 

    if i == 3:  
        axs[i].set_yticklabels([0, 0.05], fontsize=45) 

    if i == 4:  
        axs[i].set_yticklabels([0, 0.1], fontsize=45) 
        
# Colorbar customization
cbar_ax = fig.add_axes([0.91, 0.11, 0.025, 0.77])  
norm = BoundaryNorm(boundaries=np.arange(1998, 2025, 1), ncolors=cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
major_ticks = np.arange(1998, 2025, 4)
cbar.set_ticks(major_ticks)  
cbar.set_ticklabels(major_ticks)
cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
cbar.ax.tick_params(labelsize=45, direction='in', which='major', length=13, width=2)

plt.subplots_adjust(hspace=0.6)
plt.show()


outfile = r'E:\...\Figures\Fig_3\Fig_3i.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3j (Bloom Duration)
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=plt.cm.Spectral_r

cs_1= axs.pcolormesh(LON_NA, LAT_NA, np.nanmean(bloom_dur_NA, axis=2), vmin=50, vmax=120, cmap=cmap, transform=proj)
cs_2= axs.pcolormesh(LON_SA, LAT_SA, np.nanmean(bloom_dur_SA, axis=2), vmin=50, vmax=120, cmap=cmap, transform=proj)
cs_3= axs.pcolormesh(LON_AL, LAT_AL, np.nanmean(bloom_dur_AL, axis=2), vmin=50, vmax=120, cmap=cmap, transform=proj)
cs_4= axs.pcolormesh(LON_BAL, LAT_BAL, np.nanmean(bloom_dur_BAL, axis=2), vmin=50, vmax=120, cmap=cmap, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.0f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([50, 60, 70, 80, 90, 100, 110, 120])

cbar.set_label(r'[days]', fontsize=40)
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
axs.set_title('BDur', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the           [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, np.nanmean(bloom_dur_CAN, axis=2), vmin=50, vmax=120, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3j.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3k (Bloom Duration 2011-2023 minus 1998-2010)
dur_diff_SA = np.nanmean(bloom_dur_SA[:,:,13:26], axis=2) - np.nanmean(bloom_dur_SA[:,:,0:13], axis=2)
dur_diff_AL = np.nanmean(bloom_dur_AL[:,:,13:26], axis=2) - np.nanmean(bloom_dur_AL[:,:,0:13], axis=2)
dur_diff_BAL = np.nanmean(bloom_dur_BAL[:,:,13:26], axis=2) - np.nanmean(bloom_dur_BAL[:,:,0:13], axis=2)
dur_diff_NA = np.nanmean(bloom_dur_NA[:,:,13:26], axis=2) - np.nanmean(bloom_dur_NA[:,:,0:13], axis=2)
dur_diff_CAN = np.nanmean(bloom_dur_CAN[:,:,13:26], axis=2) - np.nanmean(bloom_dur_CAN[:,:,0:13], axis=2)

fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.delta
norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)

cs_1= axs.pcolormesh(LON_NA, LAT_NA, dur_diff_NA, cmap=cmap, norm=norm, transform=proj)
cs_2= axs.pcolormesh(LON_SA, LAT_SA, dur_diff_SA, cmap=cmap, norm=norm, transform=proj)
cs_3= axs.pcolormesh(LON_AL, LAT_AL, dur_diff_AL, cmap=cmap, norm=norm, transform=proj)
cs_4= axs.pcolormesh(LON_BAL, LAT_BAL, dur_diff_BAL, cmap=cmap, norm=norm, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.0f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([-30, -20, -10, 0, 10, 20, 30])

cbar.set_label(r'[days]', fontsize=40)
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
axs.set_title('BDur 2011-2023 minus 1998-2010', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the           [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, dur_diff_CAN, cmap=cmap, norm=norm, transform=proj)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3k.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')


## Fig. 3l (Density Distributions Bloom Duration)
years = np.arange(1998, 2024)

fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
data_demarcations = [bloom_dur_NA, bloom_dur_AL, bloom_dur_CAN, bloom_dur_SA, bloom_dur_BAL]
labels = ['North Atlantic (NA)', 'SoG and Alboran Sea (AL)', 'Canary (CAN)', 'South Atlantic (SA)',  'Levantine-Balearic (BAL)']

cmap = plt.cm.Spectral_r
colors = cmap(np.linspace(0, 1, len(years)))

for i, data in enumerate(data_demarcations):
    for j, year in enumerate(years):
        data_flat = data[:, :, j].ravel()  
        
        data_flat = data_flat[~np.isnan(data_flat)]
        
        if len(data_flat) > 0:
            sns.kdeplot(data_flat, color=colors[j], linewidth=2, ax=axs[i])
    
    axs[i].set_xlim(15, 175)  
    axs[i].set_title(labels[i], fontsize=45)  
    axs[i].set_xlabel(r'BDur [days]', fontsize=45)
    axs[i].set_ylabel('')
    

    axs[i].set_xticks([25, 50, 75, 100, 125, 150, 175])
    axs[i].set_xticklabels([25, 50, 75, 100, 125, 150, 175], fontsize=45)  
    # axs[i].set_yticks([0, 2.5])
    # axs[i].set_yticklabels([0, 2.5], fontsize=45)  
    
    axs[i].tick_params(axis='both', which='major', labelsize=45)  

    if i == 0:  
        axs[i].set_yticklabels([0, 0.06], fontsize=45) 

    if i == 1:  
        axs[i].set_yticklabels([0, 0.03], fontsize=45) 

    if i == 2:  
        axs[i].set_ylabel('Kernel Density', fontsize=45, labelpad=5)  
        axs[i].set_yticklabels([0, 0.06], fontsize=45) 

    if i == 3:  
        axs[i].set_yticklabels([0, 0.06], fontsize=45) 

    if i == 4:  
        axs[i].set_yticklabels([0, 0.1], fontsize=45) 
        
# Colorbar customization
cbar_ax = fig.add_axes([0.91, 0.11, 0.025, 0.77])  
norm = BoundaryNorm(boundaries=np.arange(1998, 2025, 1), ncolors=cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
major_ticks = np.arange(1998, 2025, 4)
cbar.set_ticks(major_ticks)  
cbar.set_ticklabels(major_ticks)
cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
cbar.ax.tick_params(labelsize=45, direction='in', which='major', length=13, width=2)

plt.subplots_adjust(hspace=0.6)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3l.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3m (Bloom Cumulative Chl-a)
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=plt.cm.Spectral_r
norm=LogNorm(vmin=40, vmax=400)

cs_1= axs.pcolormesh(LON_NA, LAT_NA, np.nanmean(bloom_total_cum_NA, axis=2), norm=norm, cmap=cmap, transform=proj)
cs_2= axs.pcolormesh(LON_SA, LAT_SA, np.nanmean(bloom_total_cum_SA, axis=2), norm=norm, cmap=cmap, transform=proj)
cs_3= axs.pcolormesh(LON_AL, LAT_AL, np.nanmean(bloom_total_cum_AL, axis=2), norm=norm, cmap=cmap, transform=proj)
cs_4= axs.pcolormesh(LON_BAL, LAT_BAL, np.nanmean(bloom_total_cum_BAL, axis=2), norm=norm, cmap=cmap, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.0f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
# cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.minorticks_off()
# cbar.set_ticks([16, 20, 30, 40, 50])
cbar.set_ticks([40, 70, 120, 200, 400])

cbar.set_label(r'$\mathrm{[mg \cdot m^{-3} \cdot days]}$', fontsize=40)
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
axs.set_title('BCumChl-a', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the           [left, bottom, width, height] values in the line above to adjust the position and size.

norm_CAN=LogNorm(vmin=16, vmax=50)
# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, np.nanmean(bloom_total_cum_CAN, axis=2), norm=norm_CAN, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3m.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3n (Bloom Cumulative Chl-a 2011-2023 minus 1998-2010)
total_cum_diff_SA = np.nanmean(bloom_total_cum_SA[:,:,13:26], axis=2) - np.nanmean(bloom_total_cum_SA[:,:,0:13], axis=2)
total_cum_diff_AL = np.nanmean(bloom_total_cum_AL[:,:,13:26], axis=2) - np.nanmean(bloom_total_cum_AL[:,:,0:13], axis=2)
total_cum_diff_BAL = np.nanmean(bloom_total_cum_BAL[:,:,13:26], axis=2) - np.nanmean(bloom_total_cum_BAL[:,:,0:13], axis=2)
total_cum_diff_NA = np.nanmean(bloom_total_cum_NA[:,:,13:26], axis=2) - np.nanmean(bloom_total_cum_NA[:,:,0:13], axis=2)
total_cum_diff_CAN = np.nanmean(bloom_total_cum_CAN[:,:,13:26], axis=2) - np.nanmean(bloom_total_cum_CAN[:,:,0:13], axis=2)


fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.delta
norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=25)

cs_1= axs.pcolormesh(LON_SA, LAT_SA, total_cum_diff_SA, cmap=cmap, norm=norm, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, total_cum_diff_AL, cmap=cmap, norm=norm, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, total_cum_diff_BAL, cmap=cmap, norm=norm, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, total_cum_diff_NA, cmap=cmap, norm=norm, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.98, format=ticker.FormatStrFormatter('%.0f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([-20, -10, 0, 10, 20])
cbar.set_label(r'$\mathrm{[mg \cdot m^{-3} \cdot days]}$', fontsize=40)
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
axs.set_title('BCumChl-a 2011-2023 minus 1998-2010', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, total_cum_diff_CAN, cmap=cmap, norm=norm, transform=proj)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3n.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3o (Density Distributions Bloom Cumulative Chl-a)
years = np.arange(1998, 2024)

fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
data_demarcations = [bloom_total_cum_NA, bloom_total_cum_AL, bloom_total_cum_CAN, bloom_total_cum_SA, bloom_total_cum_BAL]
labels = ['North Atlantic (NA)', 'SoG and Alboran Sea (AL)', 'Canary (CAN)', 'South Atlantic (SA)',  'Levantine-Balearic (BAL)']

cmap = plt.cm.Spectral_r
colors = cmap(np.linspace(0, 1, len(years)))

for i, data in enumerate(data_demarcations):
    for j, year in enumerate(years):
        data_flat = data[:, :, j].ravel()  
        
        data_flat = data_flat[~np.isnan(data_flat)]
        
        if len(data_flat) > 0:
            sns.kdeplot(data_flat, color=colors[j], linewidth=2, ax=axs[i])
    
    axs[i].set_xscale('log')  
    axs[i].set_xlim(15, 225)  
    axs[i].set_title(labels[i], fontsize=45)  
    axs[i].set_xlabel(r'BCumChl-a $\mathrm{[mg \cdot m^{-3} \cdot days]}$', fontsize=45)
    axs[i].set_ylabel('')
    
    axs[i].set_xticks([25, 50, 75, 125, 200])
    axs[i].set_xticklabels([25, 50, 75, 125, 200], fontsize=45)   
    
    axs[i].tick_params(axis='both', which='major', labelsize=45)  

    if i == 0:  
        axs[i].set_yticklabels([0, 0.03], fontsize=45) 

    if i == 1:  
        axs[i].set_yticklabels([0, 2.5], fontsize=45) 

    if i == 2:  
        axs[i].set_ylabel('Kernel Density', fontsize=45, labelpad=44)  
        axs[i].set_yticklabels([0, 5], fontsize=45) 

    if i == 3:  
        axs[i].set_yticklabels([0, 2.5], fontsize=45) 

    if i == 4:  
        axs[i].set_yticklabels([0, 5], fontsize=45) 
        
# Colorbar customization
cbar_ax = fig.add_axes([0.91, 0.11, 0.025, 0.77])  
norm = BoundaryNorm(boundaries=np.arange(1998, 2025, 1), ncolors=cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
major_ticks = np.arange(1998, 2025, 4)
cbar.set_ticks(major_ticks)  
cbar.set_ticklabels(major_ticks)
cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
cbar.ax.tick_params(labelsize=45, direction='in', which='major', length=13, width=2)

plt.subplots_adjust(hspace=0.6)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3o.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3p (Seasonal Cycle Reproducibility)
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=plt.cm.Spectral_r
vmin, vmax = 0.1, 0.9

cs_1= axs.pcolormesh(LON_SA, LAT_SA, np.nanmean(SCR_SA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, np.nanmean(SCR_AL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, np.nanmean(SCR_BAL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, np.nanmean(SCR_NA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])
cbar.set_label(r'[r]', fontsize=40)

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
axs.set_title('SCR', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, np.nanmean(SCR_CAN, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3p.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 3q (SCR 2011-2023 minus 1998-2010)
SCR_diff_SA = (np.nanmean(SCR_SA[:,:,13:26], axis=2) - np.nanmean(SCR_SA[:,:,0:13], axis=2))
SCR_diff_AL = (np.nanmean(SCR_AL[:,:,13:26], axis=2) - np.nanmean(SCR_AL[:,:,0:13], axis=2))
SCR_diff_BAL = (np.nanmean(SCR_BAL[:,:,13:26], axis=2) - np.nanmean(SCR_BAL[:,:,0:13], axis=2))
SCR_diff_NA = (np.nanmean(SCR_NA[:,:,13:26], axis=2) - np.nanmean(SCR_NA[:,:,0:13], axis=2))
SCR_diff_CAN = (np.nanmean(SCR_CAN[:,:,13:26], axis=2) - np.nanmean(SCR_CAN[:,:,0:13], axis=2))


fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='black', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(40)

cmap=cm.cm.delta
norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)

cs_1= axs.pcolormesh(LON_SA, LAT_SA, SCR_diff_SA, cmap=cmap, norm=norm, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, SCR_diff_AL, cmap=cmap, norm=norm, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, SCR_diff_BAL, cmap=cmap, norm=norm, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, SCR_diff_NA, cmap=cmap, norm=norm, transform=proj)

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


cbar = plt.colorbar(cs_1, shrink=0.96, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
# cbar.set_ticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
cbar.set_label(r'[r]', fontsize=40)
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
axs.set_title('SCR 2011-2023 minus 1998-2010', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, SCR_diff_CAN, cmap=cmap, norm=norm, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3q.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')


## Fig. 3r (Density Distributions SCR)
years = np.arange(1998, 2024)

fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
data_demarcations = [SCR_NA, SCR_AL, SCR_CAN, SCR_SA, SCR_BAL]
labels = ['North Atlantic (NA)', 'SoG and Alboran Sea (AL)', 'Canary (CAN)', 'South Atlantic (SA)',  'Levantine-Balearic (BAL)']

cmap = plt.cm.Spectral_r
colors = cmap(np.linspace(0, 1, len(years)))

for i, data in enumerate(data_demarcations):
    for j, year in enumerate(years):
        data_flat = data[:, :, j].ravel()  
        
        data_flat = data_flat[~np.isnan(data_flat)]
        
        if len(data_flat) > 0:
            sns.kdeplot(data_flat, color=colors[j], linewidth=2, ax=axs[i])
    
    axs[i].set_xlim(-0.2, 1)  
    axs[i].set_title(labels[i], fontsize=45)  
    axs[i].set_xlabel(r'SCR $\mathrm{[r]}$', fontsize=45)
    axs[i].set_ylabel('')
    
    axs[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[i].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=45)   
    
    axs[i].tick_params(axis='both', which='major', labelsize=45)  

    if i == 0:  
        axs[i].set_yticklabels([0, 5], fontsize=45) 

    if i == 1:  
        axs[i].set_yticklabels([0, 5], fontsize=45) 

    if i == 2:  
        axs[i].set_ylabel('Kernel Density', fontsize=45, labelpad=40)  
        axs[i].set_yticklabels([0, 5], fontsize=45) 

    if i == 3:  
        axs[i].set_yticklabels([0, 2.5], fontsize=45) 

    if i == 4:  
        axs[i].set_yticklabels([0, 5], fontsize=45) 
        
# Colorbar customization
cbar_ax = fig.add_axes([0.91, 0.11, 0.025, 0.77])  
norm = BoundaryNorm(boundaries=np.arange(1998, 2025, 1), ncolors=cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
major_ticks = np.arange(1998, 2025, 4)
cbar.set_ticks(major_ticks)  
cbar.set_ticklabels(major_ticks)
cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
cbar.ax.tick_params(labelsize=45, direction='in', which='major', length=13, width=2)

plt.subplots_adjust(hspace=0.6)
plt.show()

outfile = r'E:\...\Figures\Fig_3\Fig_3r.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')

