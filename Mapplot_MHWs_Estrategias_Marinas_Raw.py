# -*- coding: utf-8 -*-
"""

############################ Map Plot MHWs ES MAR ES  #########################

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



##Loading sst data from Datos_ESMARES.py


#######################
# Plotting SST in SMD #
#######################


fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(20)

cmap='Spectral_r'
levels=np.linspace(15, 22, num=22)

# Plot the main data
cs_1 = ax.contourf(lon_GC, lat_GC, Aver_SST_GC, levels, cmap=cmap, transform=proj, extend='max')
cs_2 = ax.contourf(lon_AL, lat_AL, Aver_SST_AL, levels, cmap=cmap, transform=proj, extend='max')
cs_3 = ax.contourf(lon_BAL, lat_BAL, Aver_SST_BAL, levels, cmap=cmap, transform=proj, extend='max')
cs_4 = ax.contourf(lon_NA, lat_NA, Aver_SST_NA, levels, cmap=cmap, transform=proj, extend='max')

# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors=['red', 'red'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['red', 'red'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['red', 'red'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['red', 'red'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2

# Plot the colorbar for main data
cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([15, 16, 17, 18, 19, 20, 21, 22])
cbar.set_label(r'[$^\circ$C]', fontsize=22)

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

plt.title(r'Averaged 1993-2022 SST', fontsize=25)


# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(lon_canarias, lat_canarias, Aver_SST_canarias, levels, cmap=cmap, transform=proj, extend='both')
# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['red', 'red'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


#Save data so far
outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figs_Test\SST_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







                        ##########################
                        ## Plotting MHW metrics ##
                        ##########################


##Load MHWs_from_MATLAB.py

                            #################
                            ###  CANARIAS ###
                            #################

#################
# MHW frequency #
#################

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_cnt_canarias), np.nanmax(MHW_cnt_canarias), num=21)
# levels=np.linspace(0.5, 1.5, num=25)

# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_cnt_CAN_SAT, levels, cmap=cmap, transform=proj, extend ='both')

# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# # cbar.set_ticks([0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45])
# cbar.set_ticks([0.5, 0.75, 1, 1.25, 1.5])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Frequency [$number$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_Freq_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #######################
# # MHW Frequency trend #
# #######################

# signif_freq_canarias = MHW_cnt_dtr_CAN_SAT
# signif_freq_canarias = np.where(signif_freq_canarias >= 0.1, np.NaN, signif_freq_canarias)
# signif_freq_canarias = np.where(signif_freq_canarias < 0.1, 1, signif_freq_canarias)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r


# # levels=np.linspace(np.nanmin(MHW_cnt_tr_canarias*10), np.nanmax(MHW_cnt_tr_canarias*10), num=20)
# levels=np.linspace(-0.5, 0.5, 25)
# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_cnt_tr_CAN_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_CAN_SAT[::4,::4], LAT_CAN_SAT[::4,::4], signif_freq_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Frequency trend [$number·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\CAN_MHW_Freq_tr.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# ################
# # MHW duration #
# ################

fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(10)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_dur_canarias), np.nanmax(MHW_dur_canarias), num=21)
levels=np.linspace(10, 16, num=25)


cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_dur_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=15) 
cbar.ax.minorticks_off()
cbar.set_ticks([10, 11, 12, 13, 14, 15, 16])


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
plt.title('MHW Duration [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_Dur_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Duration trend ##
# #######################

# signif_dur_canarias = MHW_dur_dtr_CAN_SAT
# signif_dur_canarias = np.where(signif_dur_canarias >= 0.1, np.NaN, signif_dur_canarias)
# signif_dur_canarias = np.where(signif_dur_canarias < 0.1, 1, signif_dur_canarias)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_dur_tr_canarias*10), np.nanmax(MHW_dur_tr_canarias*10), num=21)
# levels=np.linspace(-7.5, 7.5, num=25)

# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_dur_tr_CAN_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_CAN_SAT[::4,::4], LAT_CAN_SAT[::4,::4], signif_dur_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-7.5, -5, -2.5, 0, 2.5, 5, 7.5])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Duration trend [$days·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_Dur_tr_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





# ################
# # MHW Mean Int #
# ################

# fig = plt.figure(figsize=(10, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_mean_canarias), np.nanmax(MHW_mean_canarias), num=21)
# levels=np.linspace(0.5, 2.25, num=22)
# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_mean_CAN_SAT, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean Intensity [$^\circ$C]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_MeanInt_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Mean Int trend ##
# #######################

# signif_mean_canarias = MHW_mean_dtr_CAN_SAT
# signif_mean_canarias = np.where(signif_mean_canarias >= 0.1, np.NaN, signif_mean_canarias)
# signif_mean_canarias = np.where(signif_mean_canarias < 0.1, 1, signif_mean_canarias)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_mean_tr_canarias*10), np.nanmax(MHW_mean_tr_canarias*10), num=21)
# levels=np.linspace(-0.15, 0.15, num=25)

# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_mean_tr_CAN_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_CAN_SAT[::4,::4], LAT_CAN_SAT[::4,::4], signif_mean_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_MeanInt_tr_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


# ################
# # MHW Max Int ##
# ################

# fig = plt.figure(figsize=(10, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd

# levels=np.linspace(0.75, 2.5, num=22)

# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_max_CAN_SAT, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Max Intensity [$^\circ$C]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_MaxInt_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Max Int trend ###
# #######################

# signif_max_canarias = MHW_max_dtr_CAN_SAT
# signif_max_canarias = np.where(signif_max_canarias >= 0.1, np.NaN, signif_max_canarias)
# signif_max_canarias = np.where(signif_max_canarias < 0.1, 1, signif_max_canarias)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r

# levels=np.linspace(-0.3, 0.3, num=25)

# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_max_tr_CAN_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_CAN_SAT[::4,::4], LAT_CAN_SAT[::4,::4], signif_max_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Max Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_MaxInt_tr_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





# ################
# # MHW Cum Int ##
# ################

# fig = plt.figure(figsize=(10, 10))
# #change color bar to proper values in Atlantic Tropic

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_cum_canarias), np.nanmax(MHW_cum_canarias), num=21)
# levels=np.linspace(10, 25, num=25)

# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_cum_CAN_SAT, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([10, 12.5, 15, 17.5, 20, 22.5, 25])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Cumulative Intensity [$^{\circ}C\ ·  days$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_CumInt_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Cum Int trend ###
# #######################

# signif_cum_canarias = MHW_cum_dtr_CAN_SAT
# signif_cum_canarias = np.where(signif_cum_canarias >= 0.5, np.NaN, signif_cum_canarias)
# signif_cum_canarias = np.where(signif_cum_canarias < 0.5, 1, signif_cum_canarias)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r

# levels=np.linspace(-20, 20, num=25)

# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_cum_tr_CAN_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_CAN_SAT[::4,::4], LAT_CAN_SAT[::4,::4], signif_cum_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Cum Intensity trend [$^{\circ}C\ · days\ · decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_CumInt_tr_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





# ##########################
# # Total Annual MHW days ##
# ##########################

# fig = plt.figure(figsize=(10, 10))
# #change color bar to proper values in Atlantic Tropic

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd

# levels=np.linspace(20, 35, num=25)

# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_td_CAN_SAT*2, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([20, 22.5, 25, 27.5, 30, 32.5, 35])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days [$days$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_Td_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #################################
# # Total Annual MHW days trend ###
# #################################

# signif_td_canarias = MHW_td_dtr_CAN_SAT
# signif_td_canarias = np.where(signif_td_canarias >= 0.1, np.NaN, signif_td_canarias)
# signif_td_canarias = np.where(signif_td_canarias < 0.1, 1, signif_td_canarias)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # cmap=plt.cm.YlOrRd

# levels=np.linspace(-5, 5, num=25)

# cs= ax.contourf(LON_CAN_SAT, LAT_CAN_SAT, MHW_td_tr_CAN_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_CAN_SAT[::4,::4], LAT_CAN_SAT[::4,::4], signif_td_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-5, -2.5, 0, 2.5, 5])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj)
# ax.set_yticks([24,26,28,30,32], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days trend [$days·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\SATELLITE\MHW_Td_tr_CAN.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










                            #################
                            ## Golfo Cadiz ##
                            #################



#################
# MHW frequency #
#################

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_cnt_GC), np.nanmax(MHW_cnt_GC), num=21)
# levels=np.linspace(0.5, 1.25, num=22)
# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_cnt_GC_SAT, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Frequency [$number$]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_Freq.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #######################
# # MHW Frequency trend #
# #######################

# signif_freq_GC = MHW_cnt_dtr_GC_SAT
# signif_freq_GC = np.where(signif_freq_GC >= 0.1, np.NaN, signif_freq_GC)
# signif_freq_GC = np.where(signif_freq_GC < 0.1, 1, signif_freq_GC)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r

# # levels=np.linspace(np.nanmin(MHW_cnt_tr_GC*10), np.nanmax(MHW_cnt_tr_GC*10), num=20)
# levels=np.linspace(-1.2, 1.2, 25)
# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_cnt_tr_GC_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_GC_SAT[::4,::4], LAT_GC_SAT[::4,::4], signif_freq_GC[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Frequency trend [$number·decade^{-1}$]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_Freq_tr.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# ################
# # MHW duration #
# ################

# fig = plt.figure(figsize=(10, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd

# levels=np.linspace(10, 30, num=25)


# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_dur_GC_SAT, levels, cmap=cmap, transform=proj, extend ='both')

# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([10, 15, 20, 25, 30])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Duration [$days$]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_Dur.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Duration trend ##
# #######################

# signif_dur_GC = MHW_dur_dtr_GC_SAT
# signif_dur_GC = np.where(signif_dur_GC >= 0.1, np.NaN, signif_dur_GC)
# signif_dur_GC = np.where(signif_dur_GC < 0.1, 1, signif_dur_GC)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_dur_tr_GC*10), np.nanmax(MHW_dur_tr_GC*10), num=21)
# levels=np.linspace(-9, 9, num=25)

# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_dur_tr_GC_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_GC_SAT[::4,::4], LAT_GC_SAT[::4,::4], signif_dur_GC[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-9, -6, -3, 0, 3, 6, 9])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Duration trend [$days·decade^{-1}$]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_Dur_tr.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





# ################
# # MHW Mean Int #
# ################

# fig = plt.figure(figsize=(10, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# levels=np.linspace(1, 2, num=25)

# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_mean_GC_SAT, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([1, 1.25, 1.5, 1.75, 2])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean Intensity [$^\circ$C]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_MeanInt.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Mean Int trend ##
# #######################

# signif_mean_GC = MHW_mean_dtr_GC_SAT
# signif_mean_GC = np.where(signif_mean_GC >= 0.1, np.NaN, signif_mean_GC)
# signif_mean_GC = np.where(signif_mean_GC < 0.1, 1, signif_mean_GC)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_mean_tr_GC*10), np.nanmax(MHW_mean_tr_GC*10), num=21)
# levels=np.linspace(-0.4, 0.4, num=25)

# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_mean_tr_GC_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_GC_SAT[::4,::4], LAT_GC_SAT[::4,::4], signif_mean_GC[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-0.4, -0.2, 0, 0.2, 0.4])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_MeanInt_tr.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


# ################
# # MHW Max Int ##
# ################

# fig = plt.figure(figsize=(10, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd

# levels=np.linspace(1, 2.5, num=25)

# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_max_GC_SAT, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.50])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Max Intensity [$^\circ$C]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_MaxInt.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Max Int trend ###
# #######################

# signif_max_GC = MHW_max_dtr_GC_SAT
# signif_max_GC = np.where(signif_max_GC >= 0.1, np.NaN, signif_max_GC)
# signif_max_GC = np.where(signif_max_GC < 0.1, 1, signif_max_GC)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_mean_tr_GC*10), np.nanmax(MHW_mean_tr_GC*10), num=20)
# levels=np.linspace(-0.5, 0.5, num=25)

# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_max_tr_GC_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_GC_SAT[::4,::4], LAT_GC_SAT[::4,::4], signif_max_GC[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Max Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_MaxInt_tr.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# ################
# # MHW Cum Int ##
# ################

fig = plt.figure(figsize=(20, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_cum_GC), np.nanmax(MHW_cum_GC), num=25)
levels=np.linspace(15, 35, num=25)

cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_cum_GC_SAT, levels, cmap=cmap, transform=proj, extend ='both')

cbar = plt.colorbar(cs, shrink=1, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([15, 20, 25, 30, 35])
cbar.set_label(r'MHW Cum Intensity [$^{\circ}C\ · days$]', fontsize=33)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Cumulative Intensity [$^{\circ}C\ · days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Justif_Junta\GC_MHW_CumInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Cum Int trend ###
# #######################

signif_cum_GC = MHW_cum_dtr_GC_SAT
signif_cum_GC = np.where(signif_cum_GC >= 0.5, np.NaN, signif_cum_GC)
signif_cum_GC = np.where(signif_cum_GC < 0.5, 1, signif_cum_GC)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.RdYlBu_r
# levels=np.linspace(np.nanmin(MHW_cum_tr_GC*10), np.nanmax(MHW_cum_tr_GC*10), num=25)
levels=np.linspace(-12, 12, num=25)

cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_cum_tr_GC_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_GC_SAT[::3,::3], LAT_GC_SAT[::3,::3], signif_cum_GC[::3,::3], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=1, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([-12, -8, -4, 0, 4, 8, 12])
cbar.set_label(r'MHW Cum Intensity trend [$^{\circ}C\ · days\ · decade^{-1}$]', fontsize=33)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Cum Intensity trend [$^{\circ}C\ · days\ · decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Justif_Junta\GC_MHW_CumInt_tr_3.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





# ##########################
# # Total Annual MHW days ##
# ##########################

# fig = plt.figure(figsize=(10, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_td_GC), np.nanmax(MHW_td_GC), num=21)
# levels=np.linspace(10, 18, num=25)

# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_td_GC_SAT, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([10, 12, 14, 16, 18])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days [$days$]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_Td.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #################################
# # Total Annual MHW days trend ###
# #################################

# signif_td_GC = MHW_td_dtr_GC_SAT
# signif_td_GC = np.where(signif_td_GC >= 0.1, np.NaN, signif_td_GC)
# signif_td_GC = np.where(signif_td_GC < 0.1, 1, signif_td_GC)

# fig = plt.figure(figsize=(10, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # cmap=plt.cm.YlOrRd

# levels=np.linspace(-15, 15, num=25)

# cs= ax.contourf(LON_GC_SAT, LAT_GC_SAT, MHW_td_tr_GC_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_GC_SAT[::4,::4], LAT_GC_SAT[::4,::4], signif_td_GC[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.66, format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.set_ticks([-15, -10, -5, 0, 5, 10, 15])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #GC
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-7.5, -6.5, -5.5], crs=proj)
# ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days trend [$days·decade^{-1}$]', fontsize=25)


# outfile = r'C:\Users\Manuel\Desktop\MHW_metrics_1993_2022\SATELLITE\GC_MHW_Td_tr.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




#                                #############
#                                ## Alboran ##
#                                #############



# #################
# # MHW frequency #
# #################

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(16)

cmap=plt.cm.YlOrRd
levels=np.linspace(0.5, 1.5, num=25)

cs= ax.contourf(LON_AL_SAT, LAT_AL_SAT, MHW_cnt_AL_SAT, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.72, extend ='both', format=ticker.FormatStrFormatter('%.2f'))

cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=24) 
cbar.ax.minorticks_off()
cbar.set_ticks([0.5, 0.75, 1, 1.25, 1.5])
cbar.set_label(r'MHW Frequency [$number$]', fontsize=25)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj) 
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Frequency [$number$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Freq_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #######################
# # MHW Frequency trend #
# #######################

signif_freq_AL = MHW_cnt_dtr_AL_SAT
signif_freq_AL = np.where(signif_freq_AL >= 0.05, np.NaN, signif_freq_AL)
signif_freq_AL = np.where(signif_freq_AL < 0.05, 1, signif_freq_AL)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(16)

cmap=plt.cm.RdYlBu_r

# levels=np.linspace(np.nanmin(MHW_cnt_tr_AL*10), np.nanmax(MHW_cnt_tr_AL*10), num=20)
levels=np.linspace(-1.25, 1.25, 21)
cs= ax.contourf(LON_AL_SAT, LAT_AL_SAT, MHW_cnt_tr_AL_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_AL_SAT[::2,::2], LAT_AL_SAT[::2,::2], signif_freq_AL[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.72, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
# cbar.set_ticks([-0.5, 0, 0.5, 1, 1.5])
# cbar.set_label(r'MHW Frequency trend [$number·decade^{-1}$]', fontsize=25)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj) 
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Frequency trend [$number·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Freq_tr_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# ################
# # MHW duration #
# ################

fig = plt.figure(figsize=(20, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(16)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_dur_AL_SAT), np.nanmax(MHW_dur_BAL_SAT), num=21)
levels=np.linspace(10, 20, num=21)


cs= ax.contourf(LON_AL_SAT, LAT_AL_SAT, MHW_dur_AL_SAT, levels, cmap=cmap, transform=proj, extend ='both')

cbar = plt.colorbar(cs, shrink=0.72, extend ='both')
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
# cbar.set_ticks([10, 15, 20, 25, 30])

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj)
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Duration [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Dur_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Duration trend ##
# #######################

signif_dur_AL = MHW_dur_dtr_AL_SAT
signif_dur_AL = np.where(signif_dur_AL >= 0.5, np.NaN, signif_dur_AL)
signif_dur_AL = np.where(signif_dur_AL < 0.5, 1, signif_dur_AL)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(16)

cmap=plt.cm.RdYlBu_r
# levels=np.linspace(np.nanmin(MHW_dur_tr_AL*10), np.nanmax(MHW_dur_tr_AL*10), num=21)
levels=np.linspace(-12, 12, num=25)

cs= ax.contourf(LON_AL_SAT, LAT_AL_SAT, MHW_dur_tr_AL_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_AL_SAT[::2,::2], LAT_AL_SAT[::2,::2], signif_dur_AL[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.72, extend ='both')
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-12, -8, -4, 0, 4, 8, 12])

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj) 
# plt.title('MHW Duration trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Dur_tr_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





# ################
# # MHW Mean Int #
# ################

# fig = plt.figure(figsize=(20, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_mean_AL), np.nanmax(MHW_mean_AL), num=21)
# # levels=np.linspace(0.7, 2, num=21)

# cs= ax.contourf(LON_AL, LAT_AL, MHW_mean_AL, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.72, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# # cbar.set_label(r'[$days$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
# ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj) 
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean Intensity [$^\circ$C]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MeanInt_AL.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Mean Int trend ##
# #######################

# signif_mean_AL = MHW_mean_dtr_AL
# signif_mean_AL = np.where(signif_mean_AL >= 0.05, np.NaN, signif_mean_AL)
# signif_mean_AL = np.where(signif_mean_AL < 0.05, 1, signif_mean_AL)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_mean_tr_AL*10), np.nanmax(MHW_mean_tr_AL*10), num=21)
# levels=np.linspace(-0.4, 0.4, num=21)

# cs= ax.contourf(LON_AL, LAT_AL, MHW_mean_tr_AL*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_AL[::2,::2], LAT_AL[::2,::2], signif_mean_AL[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.72, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# # cbar.set_label(r'Frequency [$number$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
# ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj) 
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MeanInt_tr_AL.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


# ################
# # MHW Max Int ##
# ################

fig = plt.figure(figsize=(20, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(16)

cmap=plt.cm.YlOrRd

levels=np.linspace(1, 3, num=21)

cs= ax.contourf(LON_AL_SAT, LAT_AL_SAT, MHW_max_AL_SAT, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.72, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([1, 1.5, 2, 2.5, 3])
# cbar.set_label(r'MHW Max Intensity [$^\circ$C]', fontsize=20)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj) 
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Max Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MaxInt_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Max Int trend ###
# #######################

signif_max_AL = MHW_max_dtr_AL_SAT
signif_max_AL = np.where(signif_max_AL >= 0.05, np.NaN, signif_max_AL)
signif_max_AL = np.where(signif_max_AL < 0.05, 1, signif_max_AL)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(16)

cmap=plt.cm.RdYlBu_r
# levels=np.linspace(np.nanmin(MHW_mean_tr_AL*10), np.nanmax(MHW_mean_tr_AL*10), num=20)
levels=np.linspace(-.8, .8, num=25)

cs= ax.contourf(LON_AL_SAT, LAT_AL_SAT, MHW_max_tr_AL_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_AL_SAT[::2,::2], LAT_AL_SAT[::2,::2], signif_max_AL[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.72, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])
# cbar.set_label(r'MHW Max intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=33)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj)




outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MaxInt_tr_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# ################
# # MHW Cum Int ##
# ################

fig = plt.figure(figsize=(20, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(16)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_cum_AL), np.nanmax(MHW_cum_AL), num=25)
levels=np.linspace(15, 35, num=25)

cs= ax.contourf(LON_AL_SAT, LAT_AL_SAT, MHW_cum_AL_SAT, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.72, extend ='both', format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([15, 20, 25, 30, 35, 40, 45, 50])
# cbar.set_label(r'MHW Cumulative Intensity [$^\circ$C · days]', fontsize=33)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj) 

# plt.title('MHW Cumulative Intensity [$^\circ$C · days]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Justif_Junta\MHW_CumInt_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Cum Int trend ###
# #######################

signif_cum_AL = MHW_cum_dtr_AL_SAT
signif_cum_AL = np.where(signif_cum_AL >= 0.5, np.NaN, signif_cum_AL)
signif_cum_AL = np.where(signif_cum_AL < 0.5, 1, signif_cum_AL)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(16)

cmap=plt.cm.RdYlBu_r

levels=np.linspace(-12, 12, num=25)

cs= ax.contourf(LON_AL_SAT, LAT_AL_SAT, MHW_cum_tr_AL_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_AL_SAT[::2,::2], LAT_AL_SAT[::2,::2], signif_cum_AL[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.72, extend ='both', format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([-12, -8, -4, 0, 4, 8, 12])
# cbar.set_label(r'MHW Cum intensity trend [$^{\circ}C\ · days\ · decade^{-1}$]', fontsize=33)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj)

# plt.title('MHW Cum intensity trend [$^{\circ}C\ · days\ · decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\Figuras_Justif_Junta\MHW_CumInt_tr_AL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






# ##########################
# # Total Annual MHW days ##
# ##########################

# fig = plt.figure(figsize=(20, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_td_AL), np.nanmax(MHW_td_AL), num=21)
# levels=np.linspace(19, 34, num=21)

# cs= ax.contourf(LON_AL, LAT_AL, MHW_td_AL, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.72, extend ='both')
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([20, 22, 24, 26, 28, 30, 32, 34])
# # cbar.set_label(r'[$days$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
# ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days [$days$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Td_AL.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #################################
# # Total Annual MHW days trend ###
# #################################

# signif_td_AL = MHW_td_dtr_AL
# signif_td_AL = np.where(signif_td_AL >= 0.05, np.NaN, signif_td_AL)
# signif_td_AL = np.where(signif_td_AL < 0.05, 1, signif_td_AL)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_td_tr_AL*10), np.nanmax(MHW_td_tr_AL*10), num=21)
# levels=np.linspace(0, 28, num=21)

# cs= ax.contourf(LON_AL, LAT_AL, MHW_td_tr_AL*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_AL[::2,::2], LAT_AL[::2,::2], signif_td_AL[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.72, extend ='both')
# cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([0, 4, 8, 12, 16, 20, 24, 28])
# # cbar.set_label(r'Frequency [$number$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-6, -1.6, 35, 36.92], crs=proj)  #AL
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-6, -5, -4, -3, -2], crs=proj)
# ax.set_yticks([35.25, 35.75, 36.25, 36.75], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days trend [$days·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Td_tr_AL.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





#                                ######################
#                                ## Levantino-balear ##
#                                ######################

# #################
# # MHW frequency #
# #################

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.YlOrRd
levels=np.linspace(0.5, 1.5, num=25)
cs= ax.contourf(LON_BAL_SAT, LAT_BAL_SAT, MHW_cnt_BAL_SAT, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=1, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([0.5, 0.75, 1, 1.25, 1.5])
cbar.set_label(r'MHW Frequency [$number$]', fontsize=33)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj) 
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Frequency [$number$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Freq_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #######################
# # MHW Frequency trend #
# #######################

signif_freq_BAL = MHW_cnt_dtr_BAL_SAT
signif_freq_BAL = np.where(signif_freq_BAL >= 0.05, np.NaN, signif_freq_BAL)
signif_freq_BAL = np.where(signif_freq_BAL < 0.05, 1, signif_freq_BAL)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.RdYlBu_r

# levels=np.linspace(np.nanmin(MHW_cnt_tr_BAL*10), np.nanmax(MHW_cnt_tr_BAL*10), num=20)
levels=np.linspace(-1.25, 1.25, 21)
cs= ax.contourf(LON_BAL_SAT, LAT_BAL_SAT, MHW_cnt_tr_BAL_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_BAL_SAT[::6,::6], LAT_BAL_SAT[::6,::6], signif_freq_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=1, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25])
cbar.set_label(r'MHW Frequency trend [$number·decade^{-1}$]', fontsize=33)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj) 
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Frequency trend [$number·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Freq_tr_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# ################
# # MHW duration #
# ################

fig = plt.figure(figsize=(20, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.YlOrRd
levels=np.linspace(np.nanmin(MHW_dur_AL_SAT), np.nanmax(MHW_dur_BAL_SAT), num=21)
# levels=np.linspace(10, 20, num=21)


cs= ax.contourf(LON_BAL_SAT, LAT_BAL_SAT, MHW_dur_BAL_SAT, levels, cmap=cmap, transform=proj, extend ='both')

cbar = plt.colorbar(cs, shrink=1, extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([10, 15, 20, 25, 30])
cbar.set_label(r'MHW Duration [$days$]', fontsize=33)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj)
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Duration [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Dur_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Duration trend ##
# #######################

signif_dur_BAL = MHW_dur_dtr_BAL_SAT
signif_dur_BAL = np.where(signif_dur_BAL >= 0.5, np.NaN, signif_dur_BAL)
signif_dur_BAL = np.where(signif_dur_BAL < 0.5, 1, signif_dur_BAL)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.RdYlBu_r
# levels=np.linspace(np.nanmin(MHW_dur_tr_BAL*10), np.nanmax(MHW_dur_tr_BAL*10), num=21)
levels=np.linspace(-12, 12, num=25)

cs= ax.contourf(LON_BAL_SAT, LAT_BAL_SAT, MHW_dur_tr_BAL_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_BAL_SAT[::6,::6], LAT_BAL_SAT[::6,::6], signif_dur_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=1, extend ='both')
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([-12, -8, -4, 0, 4, 8, 12])
cbar.set_label(r'MHW Duration trend [$days·decade^{-1}$]', fontsize=33)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj) 
# plt.title(r'MHW Duration trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Dur_tr_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





# ################
# # MHW Mean Int #
# ################

# fig = plt.figure(figsize=(20, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_mean_BAL), np.nanmax(MHW_mean_BAL), num=22)
# levels=np.linspace(1, 2, num=22)

# cs= ax.contourf(LON_BAL, LAT_BAL, MHW_mean_BAL, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=1, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# # cbar.set_label(r'[$days$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6], crs=proj) 
# ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj) 
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean Intensity [$^\circ$C]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MeanInt_BAL.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Mean Int trend ##
# #######################

# signif_mean_BAL = MHW_mean_dtr_BAL
# signif_mean_BAL = np.where(signif_mean_BAL >= 0.05, np.NaN, signif_mean_BAL)
# signif_mean_BAL = np.where(signif_mean_BAL < 0.05, 1, signif_mean_BAL)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_mean_tr_BAL*10), np.nanmax(MHW_mean_tr_BAL*10), num=21)
# levels=np.linspace(-0.2, 0.2, num=25)

# cs= ax.contourf(LON_BAL, LAT_BAL, MHW_mean_tr_BAL*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_BAL[::4,::4], LAT_BAL[::4,::4], signif_mean_BAL[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=1, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6], crs=proj) 
# ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj) 
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MeanInt_tr_BAL.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


# ################
# # MHW Max Int ##
# ################

fig = plt.figure(figsize=(20, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_max_BAL), np.nanmax(MHW_max_BAL), num=21)
levels=np.linspace(1, 3, num=21)

cs= ax.contourf(LON_BAL_SAT, LAT_BAL_SAT, MHW_max_BAL_SAT, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=1, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([1, 1.5, 2, 2.5, 3])
cbar.set_label(r'MHW Max Intensity [$^\circ$C]', fontsize=33)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)  
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj) 
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Max Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MaxInt_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Max Int trend ###
# #######################

signif_max_BAL = MHW_max_dtr_BAL_SAT
signif_max_BAL = np.where(signif_max_BAL >= 0.5, np.NaN, signif_max_BAL)
signif_max_BAL = np.where(signif_max_BAL < 0.5, 1, signif_max_BAL)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.RdYlBu_r
levels=np.linspace(-.8, .8, num=25)

cs= ax.contourf(LON_BAL_SAT, LAT_BAL_SAT, MHW_max_tr_BAL_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_BAL_SAT[::6,::6], LAT_BAL_SAT[::6,::6], signif_max_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=1, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])
cbar.set_label(r'MHW Max Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=33)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)  #BAL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj)



outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MaxInt_tr_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





#################
## MHW Cum Int ##
#################

fig = plt.figure(figsize=(20, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_cum_BAL), np.nanmax(MHW_cum_BAL), num=25)
levels=np.linspace(15, 50, num=22)

cs= ax.contourf(LON_BAL_SAT, LAT_BAL_SAT, MHW_cum_BAL_SAT, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=1, extend ='both', format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
# cbar.set_ticks([25, 30, 35, 40, 45, 50])
cbar.set_label(r'MHW Cum Intensity [$^{\circ}C\ · days$]', fontsize=33)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)  
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj) 

# plt.title('MHW Cumulative Intensity [$^{\circ}C\ · days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_CumInt_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Cum Int trend ###
# #######################

signif_cum_BAL = MHW_cum_dtr_BAL_SAT
signif_cum_BAL = np.where(signif_cum_BAL >= 0.5, np.NaN, signif_cum_BAL)
signif_cum_BAL = np.where(signif_cum_BAL < 0.5, 1, signif_cum_BAL)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(22)

cmap=plt.cm.RdYlBu_r
# levels=np.linspace(np.nanmin(MHW_cum_tr_BAL*10), np.nanmax(MHW_cum_tr_BAL*10), num=20)
levels=np.linspace(-20, 20, num=25)

cs= ax.contourf(LON_BAL_SAT, LAT_BAL_SAT, MHW_cum_tr_BAL_SAT*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_BAL_SAT[::6,::6], LAT_BAL_SAT[::6,::6], signif_cum_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=1, extend ='both', format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=33) 
cbar.ax.minorticks_off()
cbar.set_ticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
cbar.set_label(r'MHW Cum Intensity trend [$^{\circ}C\ · days\ · decade^{-1}$]', fontsize=33)

ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)  #AL
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xticks([-2, 0, 2, 4, 6], crs=proj) 
ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj)
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
# plt.title('MHW Cum intensity trend [$^{\circ}C\ · days\ · decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_CumInt_tr_BAL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





# ##########################
# # Total Annual MHW days ##
# ##########################

# fig = plt.figure(figsize=(20, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_td_BAL), np.nanmax(MHW_td_BAL), num=22)
# levels=np.linspace(20, 45, num=21)

# cs= ax.contourf(LON_BAL, LAT_BAL, MHW_td_BAL, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=1, extend ='both')
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([20, 25, 30, 35, 40, 45])
# # cbar.set_label(r'[$days$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6], crs=proj) 
# ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days [$days$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Td_BAL.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #################################
# # Total Annual MHW days trend ###
# #################################

# signif_td_BAL = MHW_td_dtr_BAL
# signif_td_BAL = np.where(signif_td_BAL >= 0.05, np.NaN, signif_td_BAL)
# signif_td_BAL = np.where(signif_td_BAL < 0.05, 1, signif_td_BAL)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_td_tr_BAL*10), np.nanmax(MHW_td_tr_BAL*10), num=21)
# levels=np.linspace(0, 40, num=21)

# cs= ax.contourf(LON_BAL, LAT_BAL, MHW_td_tr_BAL*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_BAL[::4,::4], LAT_BAL[::4,::4], signif_td_BAL[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=1, extend ='both')
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# # cbar.ax.get_yaxis().set_ticks([0, 4, 8, 12, 16, 20, 24, 28])
# # cbar.set_label(r'Frequency [$number$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-2.4, 6.5, 35.9, 43], crs=proj)  
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6], crs=proj) 
# ax.set_yticks([36, 37, 38, 39, 40, 41, 42, 43], crs=proj)
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days trend [$days·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Td_tr_BAL.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





#                                ##################
#                                ## Noratlántica ##
#                                ##################

# #################
# # MHW frequency #
# #################

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_cnt_NA), np.nanmax(MHW_cnt_NA), num=22)
# levels=np.linspace(0.5, 1.5, num=21)
# cs= ax.contourf(LON_NA, LAT_NA, MHW_cnt_NA, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([0.5, 0.75, 1, 1.25, 1.5])
# # cbar.set_label(r'Frequency [$number$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj)  
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Frequency [$number$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Freq_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #######################
# # MHW Frequency trend #
# #######################

# signif_freq_NA = MHW_cnt_dtr_NA
# signif_freq_NA = np.where(signif_freq_NA >= 0.05, np.NaN, signif_freq_NA)
# signif_freq_NA = np.where(signif_freq_NA < 0.05, 1, signif_freq_NA)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r

# # levels=np.linspace(np.nanmin(MHW_cnt_tr_NA*10), np.nanmax(MHW_cnt_tr_NA*10), num=20)
# levels=np.linspace(-0.25, 0.75, 25)
# cs= ax.contourf(LON_NA, LAT_NA, MHW_cnt_tr_NA*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_NA[::4,::4], LAT_NA[::4,::4], signif_freq_NA[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# # cbar.ax.get_yaxis().set_ticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj)  
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Frequency trend [$number·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Freq_tr_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# ################
# # MHW duration #
# ################

# fig = plt.figure(figsize=(20, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_dur_NA), np.nanmax(MHW_dur_NA), num=21)
# levels=np.linspace(15, 50, num=22)


# cs= ax.contourf(LON_NA, LAT_NA, MHW_dur_NA, levels, cmap=cmap, transform=proj, extend ='both')

# cbar = plt.colorbar(cs, shrink=0.8, extend ='both')
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([15, 20, 25, 30, 35, 40, 45, 50])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj) 
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Duration [$days$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Dur_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Duration trend ##
# #######################

# signif_dur_NA = MHW_dur_dtr_NA
# signif_dur_NA = np.where(signif_dur_NA >= 0.5, np.NaN, signif_dur_NA)
# signif_dur_NA = np.where(signif_dur_NA < 0.5, 1, signif_dur_NA)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_dur_tr_NA*10), np.nanmax(MHW_dur_tr_NA*10), num=21)
# levels=np.linspace(-14, 14, num=25)

# cs= ax.contourf(LON_NA, LAT_NA, MHW_dur_tr_NA*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_NA[::4,::4], LAT_NA[::4,::4], signif_dur_NA[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both')
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# # cbar.ax.get_yaxis().set_ticks([-12, -8, -4, 0, 4, 8, 12])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj)  
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Duration trend [$days·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Dur_tr_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





# ################
# # MHW Mean Int #
# ################

# fig = plt.figure(figsize=(20, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_mean_NA), np.nanmax(MHW_mean_NA), num=22)
# levels=np.linspace(0.5, 1.5, num=25)

# cs= ax.contourf(LON_NA, LAT_NA, MHW_mean_NA, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([0.5, 0.75, 1, 1.25, 1.5])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj)  
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean Intensity [$^\circ$C]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MeanInt_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Mean Int trend ##
# #######################

# signif_mean_NA = MHW_mean_dtr_NA
# signif_mean_NA = np.where(signif_mean_NA >= 0.05, np.NaN, signif_mean_NA)
# signif_mean_NA = np.where(signif_mean_NA < 0.05, 1, signif_mean_NA)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_mean_tr_NA*10), np.nanmax(MHW_mean_tr_NA*10), num=21)
# levels=np.linspace(-0.15, 0.15, num=25)

# cs= ax.contourf(LON_NA, LAT_NA, MHW_mean_tr_NA*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_NA[::4,::4], LAT_NA[::4,::4], signif_mean_NA[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj)  
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Mean intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MeanInt_tr_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


# ################
# # MHW Max Int ##
# ################

# fig = plt.figure(figsize=(20, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_max_NA), np.nanmax(MHW_max_NA), num=21)
# levels=np.linspace(0.5, 2, num=25)

# cs= ax.contourf(LON_NA, LAT_NA, MHW_max_NA, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)  
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj)  
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Max Intensity [$^\circ$C]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MaxInt_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Max Int trend ###
# #######################

# signif_max_NA = MHW_max_dtr_NA
# signif_max_NA = np.where(signif_max_NA >= 0.05, np.NaN, signif_max_NA)
# signif_max_NA = np.where(signif_max_NA < 0.05, 1, signif_max_NA)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # levels=np.linspace(np.nanmin(MHW_mean_tr_NA*10), np.nanmax(MHW_mean_tr_NA*10), num=20)
# levels=np.linspace(-0.3, 0.3, num=25)

# cs= ax.contourf(LON_NA, LAT_NA, MHW_max_tr_NA*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_NA[::4,::4], LAT_NA[::4,::4], signif_max_NA[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
# # cbar.set_label(r'Frequency [$number$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)  #AL
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj) 
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Max intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MaxInt_tr_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# ################
# # MHW Cum Int ##
# ################

# fig = plt.figure(figsize=(20, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_cum_NA), np.nanmax(MHW_cum_NA), num=21)
# levels=np.linspace(10, 60, num=21)

# cs= ax.contourf(LON_NA, LAT_NA, MHW_cum_NA, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both', format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([10, 20, 30, 40, 50, 60])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)  
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj)  
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Cumulative Intensity [$^{\circ}C\ ·days$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_CumInt_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# #######################
# # MHW Cum Int trend ###
# #######################

# signif_cum_NA = MHW_cum_dtr_NA
# signif_cum_NA = np.where(signif_cum_NA >= 0.5, np.NaN, signif_cum_NA)
# signif_cum_NA = np.where(signif_cum_NA < 0.5, 1, signif_cum_NA)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r

# # levels=np.linspace(np.nanmin(MHW_cum_tr_NA*10), np.nanmax(MHW_cum_tr_NA*10), num=25)
# levels=np.linspace(-15, 15, num=25)

# cs= ax.contourf(LON_NA, LAT_NA, MHW_cum_tr_NA*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_NA[::4,::4], LAT_NA[::4,::4], signif_cum_NA[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both', format=ticker.FormatStrFormatter('%.0f'))
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([-15, -10, -5, 0, 5, 10, 15])
# # cbar.set_label(r'Frequency [$number$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)  #AL
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj) 
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('MHW Cum intensity trend [$^{\circ}C\ days\ · decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_CumInt_tr_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




# ##########################
# # Total Annual MHW days ##
# ##########################

# fig = plt.figure(figsize=(20, 10))
# #change color bar to proper values in Atlantic Tropic


# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
# 	label.set_fontsize(20)

# cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_td_NA), np.nanmax(MHW_td_NA), num=22)
# levels=np.linspace(15, 30, num=25)

# cs= ax.contourf(LON_NA, LAT_NA, MHW_td_NA, levels, cmap=cmap, transform=proj, extend ='both')


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both')
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([16, 18, 20, 22, 24, 26, 28, 30])

# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj) 
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days [$days$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Td_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



# #################################
# # Total Annual MHW days trend ###
# #################################

# signif_td_NA = MHW_td_dtr_NA
# signif_td_NA = np.where(signif_td_NA >= 0.05, np.NaN, signif_td_NA)
# signif_td_NA = np.where(signif_td_NA < 0.05, 1, signif_td_NA)

# fig = plt.figure(figsize=(20, 10))

# proj=ccrs.PlateCarree()#choose of projection
# #Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                     edgecolor='none', facecolor='black', linewidth=0.5)
# ax = plt.axes(projection=proj)#
# # Set tick font size
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#  	label.set_fontsize(20)

# cmap=plt.cm.RdYlBu_r
# # cmap=plt.cm.YlOrRd
# # levels=np.linspace(np.nanmin(MHW_td_tr_NA*10), np.nanmax(MHW_td_tr_NA*10), num=21)
# levels=np.linspace(-10, 20, num=25)

# cs= ax.contourf(LON_NA, LAT_NA, MHW_td_tr_NA*10, levels, cmap=cmap, transform=proj, extend ='both')
# cs2=ax.scatter(LON_NA[::4,::4], LAT_NA[::4,::4], signif_td_NA[::4,::4], color='black',linewidth=1,marker='o', alpha=0.8)


# cbar = plt.colorbar(cs, shrink=0.8, extend ='both')
# cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
# cbar.ax.minorticks_off()
# cbar.ax.get_yaxis().set_ticks([-10, -5, 0, 5, 10, 15, 20])
# # cbar.set_label(r'Frequency [$number$]', fontsize=20)
# ax.coastlines(resolution='10m', color='black', linewidth=1)
# ax.add_feature(land_10m)
# ax.add_feature(cft.BORDERS)
# ax.set_extent([-14, -1.5, 41, 47], crs=proj)  
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-14, -12, -10, -8, -6, -4, -2], crs=proj) 
# ax.set_yticks([41, 42, 43, 44, 45, 46, 47], crs=proj) 
# #ax.set_extent(img_extent, crs=proj_carr)
# #Set the title
# plt.title('Total Annual MHW days trend [$days·decade^{-1}$]', fontsize=25)


# outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Td_tr_NA.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



















                              #################
                              ## Total SMHWs ##
                              #################

#################
# MHW frequency #
#################

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
# levels=np.linspace(np.nanmin(MHW_cnt_NA), np.nanmax(MHW_cnt_NA), num=22)
levels=np.linspace(0.5, 1.5, num=25)
cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_cnt_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_cnt_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_cnt_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_cnt_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

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
cbar.set_ticks([0.5, 0.75, 1, 1.25, 1.5])
cbar.set_label(r'SMHW Frequency [$number$]', fontsize=22)
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
# axs.set_title('SMHW Frequency [$number$]', fontsize=22)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_cnt_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)



outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\SMHW_Freq_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








#######################
# MHW Frequency trend #
#######################

##Total Iberia
signif_freq_GC = MHW_cnt_dtr_GC_MODEL
signif_freq_GC = np.where(signif_freq_GC >= 0.1, np.NaN, signif_freq_GC)
signif_freq_GC = np.where(signif_freq_GC < 0.1, 1, signif_freq_GC)

signif_freq_AL = MHW_cnt_dtr_AL_MODEL
signif_freq_AL = np.where(signif_freq_AL >= 0.1, np.NaN, signif_freq_AL)
signif_freq_AL = np.where(signif_freq_AL < 0.1, 1, signif_freq_AL)

signif_freq_BAL = MHW_cnt_dtr_BAL_MODEL
signif_freq_BAL = np.where(signif_freq_BAL >= 0.1, np.NaN, signif_freq_BAL)
signif_freq_BAL = np.where(signif_freq_BAL < 0.1, 1, signif_freq_BAL)


signif_freq_NA = MHW_cnt_dtr_NA_MODEL
signif_freq_NA = np.where(signif_freq_NA >= 0.1, np.NaN, signif_freq_NA)
signif_freq_NA = np.where(signif_freq_NA < 0.1, 1, signif_freq_NA)

signif_freq_CAN = MHW_cnt_dtr_CAN_MODEL
signif_freq_CAN = np.where(signif_freq_CAN >= 0.8, np.NaN, signif_freq_CAN)
signif_freq_CAN = np.where(signif_freq_CAN < 0.5, 1, signif_freq_CAN)


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
levels=np.linspace(-1.5, 1.5, 25)


cs_1= axs.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_cnt_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=axs.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_freq_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_cnt_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=axs.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_freq_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_cnt_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=axs.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_freq_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_cnt_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=axs.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_freq_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)

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


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
cbar.set_label(r'SMHW Frequency trend [$number·decade^{-1}$]', fontsize=22)
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
# plt.title('MHW Frequency trend [$number·decade^{-1}$]', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_cnt_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_5=ax.scatter(LON_CAN_MODEL[::6,::6], LAT_CAN_MODEL[::6,::6], signif_freq_CAN[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)

# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_Freq_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










################
# MHW duration #
################

##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_cnt_NA), np.nanmax(MHW_cnt_NA), num=22)
levels=np.linspace(10, 20, num=25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_dur_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_dur_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_dur_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_dur_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs_1, shrink=0.95)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
# cbar.set_ticks([10, 15, 20, 25, 30, 35, 40])
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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
plt.title('MHW Duration [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_Dur_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_dur_canarias), np.nanmax(MHW_dur_canarias), num=21)
levels=np.linspace(10, 16, num=25)


cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_dur_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
# cbar.set_ticks([10, 11, 12, 13, 14, 15, 16])


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
plt.title('MHW Duration [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_Dur_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








#######################
# MHW Duration trend ##
#######################


##Total Iberia
signif_dur_GC = MHW_cnt_dtr_GC_MODEL
signif_dur_GC = np.where(signif_dur_GC >= 0.1, np.NaN, signif_dur_GC)
signif_dur_GC = np.where(signif_dur_GC < 0.1, 1, signif_dur_GC)

signif_dur_AL = MHW_cnt_dtr_AL_MODEL
signif_dur_AL = np.where(signif_dur_AL >= 0.1, np.NaN, signif_dur_AL)
signif_dur_AL = np.where(signif_dur_AL < 0.5, 1, signif_dur_AL)

signif_dur_BAL = MHW_cnt_dtr_BAL_MODEL
signif_dur_BAL = np.where(signif_dur_BAL >= 0.1, np.NaN, signif_dur_BAL)
signif_dur_BAL = np.where(signif_dur_BAL < 0.1, 1, signif_dur_BAL)


signif_dur_NA = MHW_cnt_dtr_NA_MODEL
signif_dur_NA = np.where(signif_dur_NA >= 0.1, np.NaN, signif_dur_NA)
signif_dur_NA = np.where(signif_dur_NA < 0.1, 1, signif_dur_NA)



fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r


levels=np.linspace(-15, 15, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_dur_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_dur_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_dur_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_dur_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_dur_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_dur_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_dur_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_dur_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-15, -10, -5, 0, 5, 10, 15])

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
plt.title('MHW Duration trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_Dur_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_dur_canarias = MHW_dur_dtr_CAN_MODEL
signif_dur_canarias = np.where(signif_dur_canarias >= 0.5, np.NaN, signif_dur_canarias)
signif_dur_canarias = np.where(signif_dur_canarias < 0.5, 1, signif_dur_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
# levels=np.linspace(np.nanmin(MHW_dur_tr_canarias*10), np.nanmax(MHW_dur_tr_canarias*10), num=21)
levels=np.linspace(-7.5, 7.5, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_dur_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_dur_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-7.5, -5, -2.5, 0, 2.5, 5, 7.5])

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
plt.title('MHW Duration trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_Dur_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




################
# MHW Mean Int #
################

##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(0.5, 2.25, num=22)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_mean_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_mean_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_mean_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_mean_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25])
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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
plt.title('MHW Mean Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_MeanInt_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_mean_canarias), np.nanmax(MHW_mean_canarias), num=21)
levels=np.linspace(0.5, 2, num=22)
cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_mean_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
# cbar.set_ticks([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25])

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
plt.title('MHW Mean Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_MeanInt_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)











#######################
# MHW Mean Int trend ##
#######################


##Total Iberia
signif_mean_GC = MHW_mean_dtr_GC_MODEL
signif_mean_GC = np.where(signif_mean_GC >= 0.1, np.NaN, signif_mean_GC)
signif_mean_GC = np.where(signif_mean_GC < 0.1, 1, signif_mean_GC)

signif_mean_AL = MHW_mean_dtr_AL_MODEL
signif_mean_AL = np.where(signif_mean_AL >= 0.1, np.NaN, signif_mean_AL)
signif_mean_AL = np.where(signif_mean_AL < 0.1, 1, signif_mean_AL)

signif_mean_BAL = MHW_mean_dtr_BAL_MODEL
signif_mean_BAL = np.where(signif_mean_BAL >= 0.1, np.NaN, signif_mean_BAL)
signif_mean_BAL = np.where(signif_mean_BAL < 0.1, 1, signif_mean_BAL)


signif_mean_NA = MHW_mean_dtr_NA_MODEL
signif_mean_NA = np.where(signif_mean_NA >= 0.1, np.NaN, signif_mean_NA)
signif_mean_NA = np.where(signif_mean_NA < 0.1, 1, signif_mean_NA)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

# levels=np.linspace(np.nanmin(MHW_cnt_tr_NA*10), np.nanmax(MHW_cnt_tr_NA*10), num=20)
levels=np.linspace(-0.4, 0.4, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_mean_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_mean_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_mean_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_mean_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_mean_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_mean_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_mean_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_mean_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
# cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])

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
plt.title('MHW Mean intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_MeanInt_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_mean_canarias = MHW_mean_dtr_CAN_MODEL
signif_mean_canarias = np.where(signif_mean_canarias >= 0.1, np.NaN, signif_mean_canarias)
signif_mean_canarias = np.where(signif_mean_canarias < 0.1, 1, signif_mean_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
# levels=np.linspace(np.nanmin(MHW_mean_tr_canarias*10), np.nanmax(MHW_mean_tr_canarias*10), num=21)
levels=np.linspace(-0.15, 0.15, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_mean_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_mean_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])

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
plt.title('MHW Mean Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_MeanInt_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










################
# MHW Max Int ##
################


##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(0.75, 2.5, num=22)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_max_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_max_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_max_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_max_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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
plt.title('MHW Max Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_MaxInt_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(0.75, 2, num=22)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_max_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])

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
plt.title('MHW Max Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_MaxInt_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










#######################
# MHW Max Int trend ###
#######################


##Total Iberia
signif_max_GC = MHW_max_dtr_GC_MODEL
signif_max_GC = np.where(signif_max_GC >= 0.1, np.NaN, signif_max_GC)
signif_max_GC = np.where(signif_max_GC < 0.1, 1, signif_max_GC)

signif_max_AL = MHW_max_dtr_AL_MODEL
signif_max_AL = np.where(signif_max_AL >= 0.1, np.NaN, signif_max_AL)
signif_max_AL = np.where(signif_max_AL < 0.1, 1, signif_max_AL)

signif_max_BAL = MHW_max_dtr_BAL_MODEL
signif_max_BAL = np.where(signif_max_BAL >= 0.1, np.NaN, signif_max_BAL)
signif_max_BAL = np.where(signif_max_BAL < 0.1, 1, signif_max_BAL)

signif_max_NA = MHW_max_dtr_NA_MODEL
signif_max_NA = np.where(signif_max_NA >= 0.1, np.NaN, signif_max_NA)
signif_max_NA = np.where(signif_max_NA < 0.1, 1, signif_max_NA)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

levels=np.linspace(-0.6, 0.6, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_max_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_max_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_max_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_max_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_max_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_max_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_max_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_max_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])

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
plt.title('MHW Max intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_MaxInt_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_max_canarias = MHW_max_dtr_CAN_MODEL
signif_max_canarias = np.where(signif_max_canarias >= 0.1, np.NaN, signif_max_canarias)
signif_max_canarias = np.where(signif_max_canarias < 0.1, 1, signif_max_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

levels=np.linspace(-0.3, 0.3, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_max_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_max_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])

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
plt.title('MHW Max Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_MaxInt_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










################
# MHW Cum Int ##
################


##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(10, 50, num=25)


cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_cum_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_cum_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_cum_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_cum_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([10, 20, 30, 40, 50])

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
plt.title('MHW Cumulative Intensity [$^{\circ}C\ days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_CumInt_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(MHW_cum_canarias), np.nanmax(MHW_cum_canarias), num=21)
levels=np.linspace(10, 25, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_cum_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([10, 12.5, 15, 17.5, 20, 22.5, 25])

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
plt.title(r'MHW Cumulative Intensity [$^{\circ}C\ ·  days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_CumInt_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







#######################
# MHW Cum Int trend ###
#######################

##Total Iberia
signif_cum_GC = MHW_cum_dtr_GC_MODEL
signif_cum_GC = np.where(signif_cum_GC >= 0.5, np.NaN, signif_cum_GC)
signif_cum_GC = np.where(signif_cum_GC < 0.5, 1, signif_cum_GC)

signif_cum_AL = MHW_cum_dtr_AL_MODEL
signif_cum_AL = np.where(signif_cum_AL >= 0.5, np.NaN, signif_cum_AL)
signif_cum_AL = np.where(signif_cum_AL < 0.5, 1, signif_cum_AL)

signif_cum_BAL = MHW_cum_dtr_BAL_MODEL
signif_cum_BAL = np.where(signif_cum_BAL >= 0.5, np.NaN, signif_cum_BAL)
signif_cum_BAL = np.where(signif_cum_BAL < 0.5, 1, signif_cum_BAL)


signif_cum_NA = MHW_cum_dtr_NA_MODEL
signif_cum_NA = np.where(signif_cum_NA >= 0.5, np.NaN, signif_cum_NA)
signif_cum_NA = np.where(signif_cum_NA < 0.5, 1, signif_cum_NA)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

levels=np.linspace(-20, 20, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_cum_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_cum_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_cum_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_cum_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_cum_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_cum_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_cum_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_cum_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])

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
plt.title('MHW Cumulative Intensity trend [$^{\circ}C\ days\ decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_CumInt_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_cum_canarias = MHW_cum_dtr_CAN_MODEL
signif_cum_canarias = np.where(signif_cum_canarias >= 0.5, np.NaN, signif_cum_canarias)
signif_cum_canarias = np.where(signif_cum_canarias < 0.5, 1, signif_cum_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

levels=np.linspace(-20, 20, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_cum_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_cum_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])

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
plt.title('MHW Cum Intensity trend [$^{\circ}C\ · days\ · decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_CumInt_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










##########################
# Total Annual MHW days ##
##########################


##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(15, 40, num=25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, np.nanmean(np.where(MHW_td_ts_GC_MODEL == 0, np.NaN, MHW_td_ts_GC_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, np.nanmean(np.where(MHW_td_ts_AL_MODEL == 0, np.NaN, MHW_td_ts_AL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, np.nanmean(np.where(MHW_td_ts_BAL_MODEL == 0, np.NaN, MHW_td_ts_BAL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, np.nanmean(np.where(MHW_td_ts_NA_MODEL == 0, np.NaN, MHW_td_ts_NA_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
# cbar.set_ticks([20, 22.5, 25, 27.5, 30, 32.5, 35])

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
plt.title(r'Total Annual MHW days [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_Td_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(20, 35, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_td_CAN_MODEL*2, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([20, 22.5, 25, 27.5, 30, 32.5, 35])

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
plt.title('Total Annual MHW days [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_Td_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








#################################
# Total Annual MHW days trend ###
#################################


##Total Iberia
signif_td_GC = MHW_td_dtr_GC_MODEL
signif_td_GC = np.where(signif_td_GC >= 0.1, np.NaN, signif_td_GC)
signif_td_GC = np.where(signif_td_GC < 0.1, 1, signif_td_GC)

signif_td_AL = MHW_td_dtr_AL_MODEL
signif_td_AL = np.where(signif_td_AL >= 0.1, np.NaN, signif_td_AL)
signif_td_AL = np.where(signif_td_AL < 0.1, 1, signif_td_AL)

signif_td_BAL = MHW_td_dtr_BAL_MODEL
signif_td_BAL = np.where(signif_td_BAL >= 0.1, np.NaN, signif_td_BAL)
signif_td_BAL = np.where(signif_td_BAL < 0.1, 1, signif_td_BAL)

signif_td_NA = MHW_td_dtr_NA_MODEL
signif_td_NA = np.where(signif_td_NA >= 0.1, np.NaN, signif_td_NA)
signif_td_NA = np.where(signif_td_NA < 0.1, 1, signif_td_NA)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
# cmap=plt.cm.YlOrRd

levels=np.linspace(-20, 20, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, MHW_td_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_td_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, MHW_td_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_td_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MHW_td_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_td_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, MHW_td_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_td_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
# cbar.set_ticks([-40, -30, -20, -10, 0, 10, 20, 30, 40])

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
plt.title('Total Annual MHW days trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_Td_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_td_canarias = MHW_td_dtr_CAN_MODEL
signif_td_canarias = np.where(signif_td_canarias >= 0.1, np.NaN, signif_td_canarias)
signif_td_canarias = np.where(signif_td_canarias < 0.1, 1, signif_td_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
# cmap=plt.cm.YlOrRd

levels=np.linspace(-5, 5, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MHW_td_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_td_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-5, -2.5, 0, 2.5, 5])

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
plt.title('Total Annual MHW days trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\MHW_Td_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




























































































                              #################
                              ## Total BMHWs ##
                              #################

##################
# BMHW frequency #
##################

##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

# cmap=plt.cm.magma_r
cmap=plt.cm.YlOrRd


levels=np.linspace(0, 1.5, num=25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_cnt_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_cnt_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_cnt_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_cnt_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')


# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
# cbar.set_label(r'[$number$]', fontsize=20)
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
plt.title('BMHW Frequency [$number$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Freq_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




##Canarias
fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd
# cmap=plt.cm.magma_r



# levels=np.linspace(np.nanmin(BMHW_cnt_canarias), np.nanmax(BMHW_cnt_canarias), num=21)
levels=np.linspace(0, 1.5, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_cnt_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
# cbar.set_ticks([0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45])
cbar.set_ticks([0, 0.25, 0.50, 0.75, 1, 1.25, 1.5])

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
plt.title('BMHW Frequency [$number$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Freq_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)









#######################
# BMHW Frequency trend #
#######################

##Total Iberia
signif_freq_GC = BMHW_cnt_dtr_GC_MODEL
signif_freq_GC = np.where(signif_freq_GC >= 0.1, np.NaN, signif_freq_GC)
signif_freq_GC = np.where(signif_freq_GC < 0.1, 1, signif_freq_GC)

signif_freq_AL = BMHW_cnt_dtr_AL_MODEL
signif_freq_AL = np.where(signif_freq_AL >= 0.1, np.NaN, signif_freq_AL)
signif_freq_AL = np.where(signif_freq_AL < 0.1, 1, signif_freq_AL)

signif_freq_BAL = BMHW_cnt_dtr_BAL_MODEL
signif_freq_BAL = np.where(signif_freq_BAL >= 0.1, np.NaN, signif_freq_BAL)
signif_freq_BAL = np.where(signif_freq_BAL < 0.1, 1, signif_freq_BAL)


signif_freq_NA = BMHW_cnt_dtr_NA_MODEL
signif_freq_NA = np.where(signif_freq_NA >= 0.1, np.NaN, signif_freq_NA)
signif_freq_NA = np.where(signif_freq_NA < 0.1, 1, signif_freq_NA)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

# levels=np.linspace(np.nanmin(BMHW_cnt_tr_NA*10), np.nanmax(BMHW_cnt_tr_NA*10), num=20)
levels=np.linspace(-1.5, 1.5, 25)
cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_cnt_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_freq_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_cnt_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_freq_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_cnt_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_freq_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_cnt_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_freq_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)


# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2



cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

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
plt.title('BMHW Frequency trend [$number·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Freq_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




##Canarias
signif_freq_canarias = BMHW_cnt_dtr_CAN_MODEL
signif_freq_canarias = np.where(signif_freq_canarias >= 0.1, np.NaN, signif_freq_canarias)
signif_freq_canarias = np.where(signif_freq_canarias < 0.1, 1, signif_freq_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r


# levels=np.linspace(np.nanmin(BMHW_cnt_tr_canarias*10), np.nanmax(BMHW_cnt_tr_canarias*10), num=20)
levels=np.linspace(-1.5, 1.5, 25)
cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_cnt_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_freq_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

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
plt.title('BMHW Frequency trend [$number·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Freq_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)









################
# BMHW duration #
################

##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

# cmap=plt.cm.magma_r
cmap=plt.cm.YlOrRd

# levels=np.linspace(np.nanmin(BMHW_cnt_NA), np.nanmax(BMHW_cnt_NA), num=22)
levels=np.linspace(10, 55, num=28)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_dur_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_dur_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_dur_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_dur_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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
plt.title('BMHW Duration [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Dur_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd
# cmap=plt.cm.magma_r

# levels=np.linspace(np.nanmin(BMHW_dur_canarias), np.nanmax(BMHW_dur_canarias), num=21)
levels=np.linspace(10, 55, num=28)


cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_dur_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([10, 15, 20, 25, 30, 35, 40, 45, 50, 55])


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
plt.title('BMHW Duration [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Dur_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








#######################
# BMHW Duration trend ##
#######################


##Total Iberia
signif_dur_GC = BMHW_cnt_dtr_GC_MODEL
signif_dur_GC = np.where(signif_dur_GC >= 0.1, np.NaN, signif_dur_GC)
signif_dur_GC = np.where(signif_dur_GC < 0.1, 1, signif_dur_GC)

signif_dur_AL = BMHW_cnt_dtr_AL_MODEL
signif_dur_AL = np.where(signif_dur_AL >= 0.1, np.NaN, signif_dur_AL)
signif_dur_AL = np.where(signif_dur_AL < 0.5, 1, signif_dur_AL)

signif_dur_BAL = BMHW_cnt_dtr_BAL_MODEL
signif_dur_BAL = np.where(signif_dur_BAL >= 0.1, np.NaN, signif_dur_BAL)
signif_dur_BAL = np.where(signif_dur_BAL < 0.1, 1, signif_dur_BAL)


signif_dur_NA = BMHW_cnt_dtr_NA_MODEL
signif_dur_NA = np.where(signif_dur_NA >= 0.1, np.NaN, signif_dur_NA)
signif_dur_NA = np.where(signif_dur_NA < 0.1, 1, signif_dur_NA)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
# cmap=cm.cm.haline_r

levels=np.linspace(-60, 60, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_dur_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_dur_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_dur_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_dur_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_dur_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_dur_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_dur_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_dur_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
# cbar.set_ticks([-15, -10, -5, 0, 5, 10, 15])

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
plt.title('BMHW Duration trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Dur_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_dur_canarias = BMHW_dur_dtr_CAN_MODEL
signif_dur_canarias = np.where(signif_dur_canarias >= 0.1, np.NaN, signif_dur_canarias)
signif_dur_canarias = np.where(signif_dur_canarias < 0.1, 1, signif_dur_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
# levels=np.linspace(np.nanmin(BMHW_dur_tr_canarias*10), np.nanmax(BMHW_dur_tr_canarias*10), num=21)
levels=np.linspace(-60, 60, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_dur_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_dur_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
# cbar.set_ticks([-15, -10, -5, 0, 5, 10, 15])

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
plt.title('BMHW Duration trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Dur_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




################
# BMHW Mean Int #
################

##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(0, 2.5, num=26)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_mean_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_mean_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_mean_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_mean_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.5, 1, 1.5, 2, 2.5])
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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
plt.title('BMHW Mean Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_MeanInt_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(BMHW_mean_canarias), np.nanmax(BMHW_mean_canarias), num=21)
levels=np.linspace(0, 2.5, num=26)
cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_mean_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.5, 1, 1.5, 2, 2.5])

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
plt.title('BMHW Mean Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_MeanInt_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)











#######################
# BMHW Mean Int trend ##
#######################


##Total Iberia
signif_mean_GC = BMHW_mean_dtr_GC_MODEL
signif_mean_GC = np.where(signif_mean_GC >= 0.1, np.NaN, signif_mean_GC)
signif_mean_GC = np.where(signif_mean_GC < 0.1, 1, signif_mean_GC)

signif_mean_AL = BMHW_mean_dtr_AL_MODEL
signif_mean_AL = np.where(signif_mean_AL >= 0.1, np.NaN, signif_mean_AL)
signif_mean_AL = np.where(signif_mean_AL < 0.1, 1, signif_mean_AL)

signif_mean_BAL = BMHW_mean_dtr_BAL_MODEL
signif_mean_BAL = np.where(signif_mean_BAL >= 0.1, np.NaN, signif_mean_BAL)
signif_mean_BAL = np.where(signif_mean_BAL < 0.1, 1, signif_mean_BAL)


signif_mean_NA = BMHW_mean_dtr_NA_MODEL
signif_mean_NA = np.where(signif_mean_NA >= 0.1, np.NaN, signif_mean_NA)
signif_mean_NA = np.where(signif_mean_NA < 0.1, 1, signif_mean_NA)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

# levels=np.linspace(np.nanmin(BMHW_cnt_tr_NA*10), np.nanmax(BMHW_cnt_tr_NA*10), num=20)
levels=np.linspace(-0.3, 0.3, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_mean_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_mean_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_mean_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_mean_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_mean_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_mean_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_mean_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_mean_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])

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
plt.title('BMHW Mean Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_MeanInt_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_mean_canarias = BMHW_mean_dtr_CAN_MODEL
signif_mean_canarias = np.where(signif_mean_canarias >= 0.1, np.NaN, signif_mean_canarias)
signif_mean_canarias = np.where(signif_mean_canarias < 0.1, 1, signif_mean_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
# levels=np.linspace(np.nanmin(BMHW_mean_tr_canarias*10), np.nanmax(BMHW_mean_tr_canarias*10), num=21)
levels=np.linspace(-0.3, 0.3, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_mean_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_mean_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30])

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
plt.title('BMHW Mean Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_MeanInt_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










################
# BMHW Max Int ##
################


##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(0, 2.5, num=26)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_max_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_max_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_max_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_max_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.5, 1, 1.5, 2, 2.5])
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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
plt.title('BMHW Max Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_MaxInt_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(0, 2.5, num=26)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_max_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.5, 1, 1.5, 2, 2.5])

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
plt.title('BMHW Max Intensity [$^\circ$C]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_MaxInt_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










#######################
# BMHW Max Int trend ###
#######################


##Total Iberia
signif_max_GC = BMHW_max_dtr_GC_MODEL
signif_max_GC = np.where(signif_max_GC >= 0.1, np.NaN, signif_max_GC)
signif_max_GC = np.where(signif_max_GC < 0.1, 1, signif_max_GC)

signif_max_AL = BMHW_max_dtr_AL_MODEL
signif_max_AL = np.where(signif_max_AL >= 0.1, np.NaN, signif_max_AL)
signif_max_AL = np.where(signif_max_AL < 0.1, 1, signif_max_AL)

signif_max_BAL = BMHW_max_dtr_BAL_MODEL
signif_max_BAL = np.where(signif_max_BAL >= 0.1, np.NaN, signif_max_BAL)
signif_max_BAL = np.where(signif_max_BAL < 0.1, 1, signif_max_BAL)

signif_max_NA = BMHW_max_dtr_NA_MODEL
signif_max_NA = np.where(signif_max_NA >= 0.1, np.NaN, signif_max_NA)
signif_max_NA = np.where(signif_max_NA < 0.1, 1, signif_max_NA)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

levels=np.linspace(-0.4, 0.4, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_max_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_max_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_max_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_max_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_max_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_max_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_max_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_max_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])

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
plt.title('BMHW Max Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_MaxInt_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_max_canarias = BMHW_max_dtr_CAN_MODEL
signif_max_canarias = np.where(signif_max_canarias >= 0.1, np.NaN, signif_max_canarias)
signif_max_canarias = np.where(signif_max_canarias < 0.1, 1, signif_max_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

levels=np.linspace(-0.4, 0.4, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_max_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_max_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])

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
plt.title('BMHW Max Intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_MaxInt_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










################
# BMHW Cum Int ##
################


##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(10, 50, num=25)


cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_cum_GC_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_cum_AL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_cum_BAL_MODEL, levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_cum_NA_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([10, 20, 30, 40, 50])

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
plt.title('BMHW Cumulative Intensity [$^{\circ}C\ days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_CumInt_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd
# levels=np.linspace(np.nanmin(BMHW_cum_canarias), np.nanmax(BMHW_cum_canarias), num=21)
levels=np.linspace(10, 50, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_cum_CAN_MODEL, levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([10, 20, 30, 40, 50])

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
plt.title('BMHW Cumulative Intensity [$^{\circ}C\ ·  days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_CumInt_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







#######################
# BMHW Cum Int trend ###
#######################

##Total Iberia
signif_cum_GC = BMHW_cum_dtr_GC_MODEL
signif_cum_GC = np.where(signif_cum_GC >= 0.1, np.NaN, signif_cum_GC)
signif_cum_GC = np.where(signif_cum_GC < 0.1, 1, signif_cum_GC)

signif_cum_AL = BMHW_cum_dtr_AL_MODEL
signif_cum_AL = np.where(signif_cum_AL >= 0.1, np.NaN, signif_cum_AL)
signif_cum_AL = np.where(signif_cum_AL < 0.1, 1, signif_cum_AL)

signif_cum_BAL = BMHW_cum_dtr_BAL_MODEL
signif_cum_BAL = np.where(signif_cum_BAL >= 0.1, np.NaN, signif_cum_BAL)
signif_cum_BAL = np.where(signif_cum_BAL < 0.1, 1, signif_cum_BAL)


signif_cum_NA = BMHW_cum_dtr_NA_MODEL
signif_cum_NA = np.where(signif_cum_NA >= 0.1, np.NaN, signif_cum_NA)
signif_cum_NA = np.where(signif_cum_NA < 0.1, 1, signif_cum_NA)

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

levels=np.linspace(-20, 20, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_cum_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_cum_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_cum_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_cum_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_cum_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_cum_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_cum_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_cum_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])

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
plt.title('BMHW Cumulative Intensity trend [$^{\circ}C\ days\ decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_CumInt_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_cum_canarias = BMHW_cum_dtr_CAN_MODEL
signif_cum_canarias = np.where(signif_cum_canarias >= 0.1, np.NaN, signif_cum_canarias)
signif_cum_canarias = np.where(signif_cum_canarias < 0.1, 1, signif_cum_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r

levels=np.linspace(-20, 20, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_cum_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_cum_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])

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
plt.title('BMHW Cum Intensity trend [$^{\circ}C\ · days\ · decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_CumInt_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










##########################
# Total Annual BMHW days ##
##########################


##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(15, 140, num=26)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, np.nanmean(np.where(BMHW_td_ts_GC_MODEL == 0, np.NaN, BMHW_td_ts_GC_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, np.nanmean(np.where(BMHW_td_ts_AL_MODEL == 0, np.NaN, BMHW_td_ts_AL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, np.nanmean(np.where(BMHW_td_ts_BAL_MODEL == 0, np.NaN, BMHW_td_ts_BAL_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, np.nanmean(np.where(BMHW_td_ts_NA_MODEL == 0, np.NaN, BMHW_td_ts_NA_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
# cbar.set_ticks([25, 26, 27, 28, 29, 30])

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
plt.title('Total Annual BMHW days [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Td_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.YlOrRd

levels=np.linspace(15, 140, num=26)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, np.nanmean(np.where(BMHW_td_ts_CAN_MODEL == 0, np.NaN, BMHW_td_ts_CAN_MODEL), axis=2), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
# cbar.set_ticks([25, 26, 27, 28, 29, 30])

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
plt.title('Total Annual BMHW days [$days$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Td_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








#################################
# Total Annual BMHW days trend ###
#################################


##Total Iberia
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

fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
levels=np.linspace(-40, 40, 25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, BMHW_td_tr_GC_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_1=ax.scatter(LON_GC_MODEL[::8,::8], LAT_GC_MODEL[::8,::8], signif_td_GC[::8,::8], color='black',linewidth=1,marker='o', alpha=0.8)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, BMHW_td_tr_AL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_2=ax.scatter(LON_AL_MODEL[::6,::6], LAT_AL_MODEL[::6,::6], signif_td_AL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, BMHW_td_tr_BAL_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_3=ax.scatter(LON_BAL_MODEL[::6,::6], LAT_BAL_MODEL[::6,::6], signif_td_BAL[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, BMHW_td_tr_NA_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
css_4=ax.scatter(LON_NA_MODEL[::6,::6], LAT_NA_MODEL[::6,::6], signif_td_NA[::6,::6], color='black',linewidth=1,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-40, -30, -20, -10, 0, 10, 20, 30, 40])

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
plt.title('Total Annual BMHW days trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Td_tr_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
signif_td_canarias = BMHW_td_dtr_CAN_MODEL
signif_td_canarias = np.where(signif_td_canarias >= 0.1, np.NaN, signif_td_canarias)
signif_td_canarias = np.where(signif_td_canarias < 0.1, 1, signif_td_canarias)

fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	label.set_fontsize(20)

cmap=plt.cm.RdYlBu_r
# cmap=plt.cm.YlOrRd

levels=np.linspace(-40, 40, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, BMHW_td_tr_CAN_MODEL*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_CAN_MODEL[::4,::4], LAT_CAN_MODEL[::4,::4], signif_td_canarias[::4,::4], color='black',linewidth=1.5,marker='o', alpha=0.8)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([-40, -30, -20, -10, 0, 10, 20, 30, 40])

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
plt.title('Total Annual BMHW days trend [$days·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_Td_tr_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)























































































                              ####################
                              ##  BMHWs - SMHWs ##
                              ####################



################################
# BMHW - SMHWs Mean Frequency ##
################################

##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)


cmap=plt.cm.RdBu_r


levels=np.linspace(-1, 1, num=25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, (BMHW_cnt_GC_MODEL - MHW_cnt_GC_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, (BMHW_cnt_AL_MODEL - MHW_cnt_AL_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, (BMHW_cnt_BAL_MODEL - MHW_cnt_BAL_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, (BMHW_cnt_NA_MODEL - MHW_cnt_NA_MODEL), levels, cmap=cmap, transform=proj, extend ='both')


# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
# cbar.set_ticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
cbar.set_label(r'[$number$]', fontsize=22)
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
plt.title('BMHW - SMHW Mean Frequency', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_Freq_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




##Canarias
fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)


cmap=plt.cm.RdBu_r


# levels=np.linspace(np.nanmin(BMHW_cnt_canarias), np.nanmax(BMHW_cnt_canarias), num=21)
levels=np.linspace(-1, 1, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, (BMHW_cnt_CAN_MODEL - MHW_cnt_CAN_MODEL), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
# cbar.set_ticks([0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45])
cbar.set_label(r'[$number$]', fontsize=22)

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
plt.title('BMHW - SMHW Mean Frequency', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_Freq_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







#############################
# BMHW - SMHW Mean Duration #
#############################

##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdBu_r

# levels=np.linspace(np.nanmin(BMHW_cnt_NA), np.nanmax(BMHW_cnt_NA), num=22)
levels=np.linspace(-30, 30, num=25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, (BMHW_dur_GC_MODEL - MHW_dur_GC_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, (BMHW_dur_AL_MODEL - MHW_dur_AL_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, (BMHW_dur_BAL_MODEL - MHW_dur_BAL_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, (BMHW_dur_NA_MODEL - MHW_dur_NA_MODEL), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95)
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-30,-20, -10, 0, 10, 20, 30])
cbar.set_label(r'[$days$]', fontsize=22)
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
plt.title('BMHW - SMHW Mean Duration', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_Dur_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdBu_r

# levels=np.linspace(np.nanmin(BMHW_dur_canarias), np.nanmax(BMHW_dur_canarias), num=21)
levels=np.linspace(-30, 30, num=25)


cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, (BMHW_dur_CAN_MODEL - MHW_dur_CAN_MODEL), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([-30, -20, -10, 0, 10, 20, 30])
cbar.set_label(r'[$days$]', fontsize=22)

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
plt.title('BMHW - SMHW Mean Duration', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_Dur_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)









###################################
## BMHW - SMHW Maximum Intensity ##
###################################


##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdBu_r

levels=np.linspace(-2, 2, num=25)

cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, (BMHW_max_GC_MODEL - MHW_max_GC_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, (BMHW_max_AL_MODEL - MHW_max_AL_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, (BMHW_max_BAL_MODEL - MHW_max_BAL_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, (BMHW_max_NA_MODEL - MHW_max_NA_MODEL), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
cbar.set_label(r'[$^\circ$C]', fontsize=22)
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
plt.title('BMHW - SMHW Max Intensity', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_MaxInt_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic


proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdBu_r

levels=np.linspace(-2, 2, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, (BMHW_max_CAN_MODEL - MHW_max_CAN_MODEL), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
cbar.set_label(r'[$^\circ$C]', fontsize=22)

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
plt.title('BMHW - SMHW Max Intensity', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_MaxInt_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)










#####################################
# BMHW - SMHW Cumulative Intensity ##
#####################################


##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdBu_r

levels=np.linspace(-30, 30, num=25)


cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, (BMHW_cum_GC_MODEL - MHW_cum_GC_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, (BMHW_cum_AL_MODEL - MHW_cum_AL_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, (BMHW_cum_BAL_MODEL - MHW_cum_BAL_MODEL), levels, cmap=cmap, transform=proj, extend ='both')
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, (BMHW_cum_NA_MODEL - MHW_cum_NA_MODEL), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

for line_2 in el_2.collections:
    line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

for line_3 in el_3.collections:
    line_3.set_zorder(2) # Set the zorder of the contour lines to 2

el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

for line_4 in el_4.collections:
    line_4.set_zorder(2) # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([-30,-20, -10, 0, 10, 20, 30])
cbar.set_label(r'[$^{\circ}C\ days$]', fontsize=22)

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
plt.title('BMHW - SMHW Cumulative Intensity', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_CumInt_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##Canarias
fig = plt.figure(figsize=(10, 10))
#change color bar to proper values in Atlantic Tropic

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap=plt.cm.RdBu_r


# levels=np.linspace(np.nanmin(BMHW_cum_canarias), np.nanmax(BMHW_cum_canarias), num=21)
levels=np.linspace(-30, 30, num=25)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, (BMHW_cum_CAN_MODEL - MHW_cum_CAN_MODEL), levels, cmap=cmap, transform=proj, extend ='both')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_ticks([-30,-20, -10, 0, 10, 20, 30])
cbar.set_label(r'[$^{\circ}C\ days$]', fontsize=22)

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
plt.title('BMHW - SMHW Cumulative Intensity', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_CumInt_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)






























































                 ##############################
                 ##  SYNCHRONY BMHWs - SMHWs ##
                 ##############################


BMHW_td_ts_GC_MODEL = np.where(BMHW_td_ts_GC_MODEL == 0, np.NaN, BMHW_td_ts_GC_MODEL)
BMHW_td_ts_AL_MODEL = np.where(BMHW_td_ts_AL_MODEL == 0, np.NaN, BMHW_td_ts_AL_MODEL)
BMHW_td_ts_BAL_MODEL = np.where(BMHW_td_ts_BAL_MODEL == 0, np.NaN, BMHW_td_ts_BAL_MODEL)
BMHW_td_ts_NA_MODEL = np.where(BMHW_td_ts_NA_MODEL == 0, np.NaN, BMHW_td_ts_NA_MODEL)
BMHW_td_ts_CAN_MODEL = np.where(BMHW_td_ts_CAN_MODEL == 0, np.NaN, BMHW_td_ts_CAN_MODEL)

MHW_td_ts_GC_MODEL = np.where(MHW_td_ts_GC_MODEL == 0, np.NaN, MHW_td_ts_GC_MODEL)
MHW_td_ts_AL_MODEL = np.where(MHW_td_ts_AL_MODEL == 0, np.NaN, MHW_td_ts_AL_MODEL)
MHW_td_ts_BAL_MODEL = np.where(MHW_td_ts_BAL_MODEL == 0, np.NaN, MHW_td_ts_BAL_MODEL)
MHW_td_ts_NA_MODEL = np.where(MHW_td_ts_NA_MODEL == 0, np.NaN, MHW_td_ts_NA_MODEL)
MHW_td_ts_CAN_MODEL = np.where(MHW_td_ts_CAN_MODEL == 0, np.NaN, MHW_td_ts_CAN_MODEL)



Synchrony_Matrix_GC = np.nanmean(np.round(BMHW_td_ts_GC_MODEL) <= np.round(MHW_td_ts_GC_MODEL), axis=2)
# Synchrony_Matrix_Normalized_GC = (Synchrony_Matrix_GC - np.nanmin(Synchrony_Matrix_GC)) / (np.nanmax(Synchrony_Matrix_GC) - np.nanmin(Synchrony_Matrix_GC))
Synchrony_Matrix_Normalized_GC = np.where(Synchrony_Matrix_GC == 0, np.NaN, Synchrony_Matrix_GC)

Synchrony_Matrix_AL = np.nanmean(np.round(BMHW_td_ts_AL_MODEL) <= np.round(MHW_td_ts_AL_MODEL), axis=2)
# Synchrony_Matrix_Normalized_AL = (Synchrony_Matrix_AL - np.nanmin(Synchrony_Matrix_AL)) / (np.nanmax(Synchrony_Matrix_AL) - np.nanmin(Synchrony_Matrix_AL))
Synchrony_Matrix_Normalized_AL = np.where(Synchrony_Matrix_AL == 0, np.NaN, Synchrony_Matrix_AL)

Synchrony_Matrix_BAL = np.nanmean(np.round(BMHW_td_ts_BAL_MODEL) <= np.round(MHW_td_ts_BAL_MODEL), axis=2)
# Synchrony_Matrix_Normalized_BAL = (Synchrony_Matrix_BAL - np.nanmin(Synchrony_Matrix_BAL)) / (np.nanmax(Synchrony_Matrix_BAL) - np.nanmin(Synchrony_Matrix_BAL))
Synchrony_Matrix_Normalized_BAL = np.where(Synchrony_Matrix_BAL == 0, np.NaN, Synchrony_Matrix_BAL)

Synchrony_Matrix_NA = np.nanmean(np.round(BMHW_td_ts_NA_MODEL) <= np.round(MHW_td_ts_NA_MODEL), axis=2)
# Synchrony_Matrix_Normalized_NA = (Synchrony_Matrix_NA - np.nanmin(Synchrony_Matrix_NA)) / (np.nanmax(Synchrony_Matrix_NA) - np.nanmin(Synchrony_Matrix_NA))
Synchrony_Matrix_Normalized_NA = np.where(Synchrony_Matrix_NA == 0, np.NaN, Synchrony_Matrix_NA)

Synchrony_Matrix_CAN = np.nanmean(np.round(BMHW_td_ts_CAN_MODEL) <= np.round(MHW_td_ts_CAN_MODEL), axis=2)
# Synchrony_Matrix_Normalized = (Synchrony_Matrix_CAN - np.nanmin(Synchrony_Matrix_CAN)) / (np.nanmax(Synchrony_Matrix_CAN) - np.nanmin(Synchrony_Matrix_CAN))
Synchrony_Matrix_Normalized_CAN = np.where(Synchrony_Matrix_CAN == 0, np.NaN, Synchrony_Matrix_CAN)





##Total Iberia
fig = plt.figure(figsize=(20, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

# cmap=plt.cm.hot_r
cmap=plt.cm.magma_r

levels=np.linspace(0, 1, num=11)


cs_1= ax.contourf(LON_GC_MODEL, LAT_GC_MODEL, Synchrony_Matrix_Normalized_GC, levels, cmap=cmap, transform=proj)
cs_2= ax.contourf(LON_AL_MODEL, LAT_AL_MODEL, Synchrony_Matrix_Normalized_AL, levels, cmap=cmap, transform=proj)
cs_3= ax.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, Synchrony_Matrix_Normalized_BAL, levels, cmap=cmap, transform=proj)
cs_4= ax.contourf(LON_NA_MODEL, LAT_NA_MODEL, Synchrony_Matrix_Normalized_NA, levels, cmap=cmap, transform=proj)

# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [-400]

# el_1 = ax.contour(lon_GC_bat, lat_GC_bat, elevation_GC, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# # ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

# for line_1 in el_1.collections:
#     line_1.set_zorder(2) # Set the zorder of the contour lines to 2

# el_2 = ax.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# # ax.clabel(el_2, inline=True, fontsize=10, fmt='%1.0f m')

# for line_2 in el_2.collections:
#     line_2.set_zorder(2) # Set the zorder of the contour lines to 2
    
# el_3 = ax.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# # ax.clabel(el_3, inline=True, fontsize=10, fmt='%1.0f m')

# for line_3 in el_3.collections:
#     line_3.set_zorder(2) # Set the zorder of the contour lines to 2

# el_4 = ax.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# # ax.clabel(el_4, inline=True, fontsize=10, fmt='%1.0f m')

# for line_4 in el_4.collections:
#     line_4.set_zorder(2) # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=22) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# cbar.set_label(r'[$^{\circ}C\ days$]', fontsize=22)

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
plt.title('BMHW & SMHW Synchrony', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_Synchrony_Total.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)
















##Canarias
fig = plt.figure(figsize=(10, 10))

proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)


cmap=plt.cm.magma_r


# levels=np.linspace(np.nanmin(BMHW_cnt_canarias), np.nanmax(BMHW_cnt_canarias), num=21)
levels=np.linspace(0, 1, num=11)

cs= ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Synchrony_Matrix_Normalized_CAN, levels, cmap=cmap, transform=proj)

# Define contour levels for elevation and plot elevation isolines
elev_levels = [-400]

el_1 = ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors='black', transform=proj, linestyles='solid', linewidths=0.75)
# ax.clabel(el_1, inline=True, fontsize=10, fmt='%1.0f m')

for line_1 in el_1.collections:
    line_1.set_zorder(2) # Set the zorder of the contour lines to 2

cbar = plt.colorbar(cs, shrink=0.67, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# cbar.set_label(r'[$number$]', fontsize=22)

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
plt.title('BMHW & SMHW Synchrony', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\Total_MHWs\MODEL\BMHW_SMHW_Synchrony_CAN.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)









