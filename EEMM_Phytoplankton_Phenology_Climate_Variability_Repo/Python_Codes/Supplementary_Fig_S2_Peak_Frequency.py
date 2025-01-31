# -*- coding: utf-8 -*-
"""

############# Supplementary Fig. S2. Bloom Peak and Frequency #################

"""

#Loading required Python modules
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import BoundaryNorm

import cmocean as cm

import cartopy.crs as ccrs
import cartopy.feature as cft

import seaborn as sns


## Load 'CHL_Phenology_metrics.py'


## Supplementary Fig. S2 ##

## Fig. S2a (Bloom Peak)
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
vmin, vmax = 50, 244

cs_1= axs.pcolormesh(LON_SA, LAT_SA, np.nanmean(bloom_peak_SA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, np.nanmean(bloom_peak_AL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, np.nanmean(bloom_peak_BAL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, np.nanmean(bloom_peak_NA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)

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
cbar.set_ticks([59.0, 91.5, 122.0, 152.5, 183.0, 213.5, 244.0])
cbar.set_ticklabels(['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])

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
axs.set_title('BPeak', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, np.nanmean(bloom_peak_CAN, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S2\Fig_S2a.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. S2b (Bloom Peak 2011-2023 minus 1998-2010)
peak_diff_SA = np.nanmean(bloom_peak_SA[:,:,13:26], axis=2) - np.nanmean(bloom_peak_SA[:,:,0:13], axis=2)
peak_diff_AL = np.nanmean(bloom_peak_AL[:,:,13:26], axis=2) - np.nanmean(bloom_peak_AL[:,:,0:13], axis=2)
peak_diff_BAL = np.nanmean(bloom_peak_BAL[:,:,13:26], axis=2) - np.nanmean(bloom_peak_BAL[:,:,0:13], axis=2)
peak_diff_NA = np.nanmean(bloom_peak_NA[:,:,13:26], axis=2) - np.nanmean(bloom_peak_NA[:,:,0:13], axis=2)
peak_diff_CAN = np.nanmean(bloom_peak_CAN[:,:,13:26], axis=2) - np.nanmean(bloom_peak_CAN[:,:,0:13], axis=2)


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

cs_1= axs.pcolormesh(LON_SA, LAT_SA, peak_diff_SA, cmap=cmap, norm=norm, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, peak_diff_AL, cmap=cmap, norm=norm, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, peak_diff_BAL, cmap=cmap, norm=norm, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, peak_diff_NA, cmap=cmap, norm=norm, transform=proj)

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
axs.set_title('BPeak 2011-2023 minus 1998-2010', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, peak_diff_CAN, cmap=cmap, norm=norm, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S2\Fig_S2b.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. S2c (Density Distributions Bloom Peak)
years = np.arange(1998, 2024)

fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
data_demarcations = [bloom_peak_NA, bloom_peak_AL, bloom_peak_CAN, bloom_peak_SA, bloom_peak_BAL]
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
    axs[i].set_xlabel(r'BPeak $\mathrm{[days]}$', fontsize=45)
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


outfile = r'E:\...\Figures\Fig_S2\Fig_S2c.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. S2d (Bloom Frequency)
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
vmin, vmax = 1.9, 4.5

cs_1= axs.pcolormesh(LON_SA, LAT_SA, np.nanmean(bloom_freq_SA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, np.nanmean(bloom_freq_AL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, np.nanmean(bloom_freq_BAL, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, np.nanmean(bloom_freq_NA, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)

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
cbar.set_ticks([2, 2.5, 3, 3.5, 4, 4.5])
cbar.set_label(r'[count]', fontsize=40)

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
axs.set_title('BFreq', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, np.nanmean(bloom_freq_CAN, axis=2), vmin=vmin, vmax=vmax, cmap=cmap, transform=proj)
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S2\Fig_S2d.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. S2e (Bloom Frequency 2011-2023 minus 1998-2010)
freq_diff_SA = np.nanmean(bloom_freq_SA[:,:,13:26], axis=2) - np.nanmean(bloom_freq_SA[:,:,0:13], axis=2)
freq_diff_AL = np.nanmean(bloom_freq_AL[:,:,13:26], axis=2) - np.nanmean(bloom_freq_AL[:,:,0:13], axis=2)
freq_diff_BAL = np.nanmean(bloom_freq_BAL[:,:,13:26], axis=2) - np.nanmean(bloom_freq_BAL[:,:,0:13], axis=2)
freq_diff_NA = np.nanmean(bloom_freq_NA[:,:,13:26], axis=2) - np.nanmean(bloom_freq_NA[:,:,0:13], axis=2)
freq_diff_CAN = np.nanmean(bloom_freq_CAN[:,:,13:26], axis=2) - np.nanmean(bloom_freq_CAN[:,:,0:13], axis=2)


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
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

cs_1= axs.pcolormesh(LON_SA, LAT_SA, freq_diff_SA, cmap=cmap, norm=norm, transform=proj)
cs_2= axs.pcolormesh(LON_AL, LAT_AL, freq_diff_AL, cmap=cmap, norm=norm, transform=proj)
cs_3= axs.pcolormesh(LON_BAL, LAT_BAL, freq_diff_BAL, cmap=cmap, norm=norm, transform=proj)
cs_4= axs.pcolormesh(LON_NA, LAT_NA, freq_diff_NA, cmap=cmap, norm=norm, transform=proj)

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

cbar = plt.colorbar(cs_1, shrink=0.98, format=ticker.FormatStrFormatter('%.1f'), pad=0.02, extend='both')
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=40)
cbar.ax.minorticks_off()
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
cbar.set_label(r'[count]', fontsize=40)
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
axs.set_title('BFreq 2011-2023 minus 1998-2010', fontsize=40)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.1235, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.pcolormesh(LON_CAN, LAT_CAN, freq_diff_CAN, cmap=cmap, norm=norm, transform=proj)

el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['#4D4D4D'], transform=proj,
                  linestyles=['solid'], linewidths=[2], zorder=2)

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)
plt.show()

outfile = r'E:\...\Figures\Fig_S2\Fig_S2e.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. S2f (Density Distributions Bloom Frequency)
years = np.arange(1998, 2024)

fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
data_demarcations = [bloom_freq_NA, bloom_freq_AL, bloom_freq_CAN, bloom_freq_SA, bloom_freq_BAL]
labels = ['North Atlantic (NA)', 'SoG and Alboran Sea (AL)', 'Canary (CAN)', 'South Atlantic (SA)',  'Levantine-Balearic (BAL)']

cmap = plt.cm.Spectral_r
colors = cmap(np.linspace(0, 1, len(years)))

for i, data in enumerate(data_demarcations):
    for j, year in enumerate(years):
        data_flat = data[:, :, j].ravel()  
        
        data_flat = data_flat[~np.isnan(data_flat)]
        
        if len(data_flat) > 0:
            sns.kdeplot(data_flat, color=colors[j], linewidth=2, ax=axs[i])
    
    axs[i].set_xlim(0.5, 5.5)  
    axs[i].set_title(labels[i], fontsize=45)  
    axs[i].set_xlabel(r'BFreq $\mathrm{[count]}$', fontsize=45)
    axs[i].set_ylabel('')
    
    axs[i].set_xticks([1, 2, 3, 4, 5])
    axs[i].set_xticklabels([1, 2, 3, 4, 5], fontsize=45)   
    
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

outfile = r'E:\...\Figures\Fig_S2\Fig_S2f.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')

