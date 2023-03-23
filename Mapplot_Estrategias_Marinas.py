# -*- coding: utf-8 -*-
"""

#################################  Map Plot ES MAR ES  ###################################

"""

import numpy as np
# import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from netCDF4 import Dataset 

from datetime import datetime as dt
import xarray as xr 
import rioxarray


import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import cmocean as cm
import cartopy.crs as ccrs
import cartopy.feature as cft
from shapely.geometry import mapping


############
# Canarias #
############

#SST
##Clipping dataset to shapefile dimension
# SST_dataset = xr.open_mfdataset('E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Canarias\*.nc', parallel=True)
# SST_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# SST_dataset.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file('E:\ICMAN-CSIC\Estrategias_Marinas\Demarcaciones_Marinas\shapes\canarias\cn.shp', crs="epsg:4326")

# clipped = SST_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Canarias/SST_Canary_clipped.nc')

#Load the previously-clipped dataset
ds = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Canarias/SST_Canary_clipped.nc')

sst = ds['analysed_sst']
lon = ds['lon']
lat = ds['lat']

# Select the time period and spatial extent of interest
start_date = '2022-01-01'
end_date = '2022-12-31'
sst = sst.sel(time=slice(start_date, end_date))


Aver_SST =sst.mean(dim=('time'))#.load() #Average SST 1982-2021
Aver_SST -= 273.15 #K to ºC


###############
# Golfo Cadiz #
###############

##SST

#Reading 2 datasets in different grids
ds_005 = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Golfo_Cadiz/cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE_1679396408276_1982_2018.nc')
ds_002 = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Golfo_Cadiz/IFREMER-ATL-SST-L4-NRT-OBS_FULL_TIME_SERIE_1679396657889_2019_2022.nc')

# # Define the new latitude and longitude grids
new_lat = np.linspace(min(ds_005.lat.min(), ds_002.lat.min()), max(ds_005.lat.max(), ds_002.lat.max()), 200)
new_lon = np.linspace(min(ds_005.lon.min(), ds_002.lon.min()), max(ds_005.lon.max(), ds_002.lon.max()), 200)

# # Interpolate each dataset to the high-resolution grid
ds_005_interp = ds_005.interp(lat=new_lat, lon=new_lon)
ds_002_interp = ds_002.interp(lat=new_lat, lon=new_lon)

# # Resample each dataset to the desired grid using nearest-neighbor interpolation
ds_005_resampled = ds_005_interp.reindex(lat=new_lat, lon=new_lon, method='nearest')
ds_002_resampled = ds_002_interp.reindex(lat=new_lat, lon=new_lon, method='nearest')

# #Merging both ds in 1 of 0.01º
ds_merged = xr.merge([ds_002_resampled, ds_005_resampled])

# #Saving new ds so far
ds_merged.to_netcdf(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Golfo_Cadiz/SST_GC_merged.nc')

##Clipping dataset to shapefile dimension
SST_dataset = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Golfo_Cadiz/SST_GC_merged.nc')
SST_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SST_dataset.rio.write_crs("epsg:4326", inplace=True)
Shapefile = gpd.read_file(r'E:\ICMAN-CSIC\Estrategias_Marinas\Demarcaciones_Marinas\shapes\sa\sa.shp', crs="epsg:4326")

clipped = SST_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
clipped.to_netcdf(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Golfo_Cadiz/SST_GC_clipped.nc')



#Load the previously-clipped dataset
ds = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Golfo_Cadiz/SST_GC_clipped.nc')

sst = ds['analysed_sst']
lon = ds['lon']
lat = ds['lat']

# Select the time period and spatial extent of interest
start_date = '2010-01-01'
end_date = '2010-01-31'
sst = sst.sel(time=slice(start_date, end_date))


Aver_SST =sst.mean(dim=('time'))#.load() #Average SST 1982-2021
Aver_SST -= 273.15 #K to ºC





#Pintar SST Estrategias Marinas

#Mean SST JJA 2022
fig = plt.figure(figsize=(20, 20))
#change color bar to proper values in Atlantic Tropic
vmin = 20
vmax = 23
#levels = np.linspace(vmin, vmax, num=256)
proj=ccrs.PlateCarree()#choose of projection
#Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
ax = plt.axes(projection=proj)#
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(20)

cmap='Spectral_r'
levels=np.linspace(Aver_SST.min(), Aver_SST.max(), num=256)
cs= ax.contourf(lon, lat, Aver_SST, levels, cmap=cmap, transform=proj, extend ='both')
# cs= ax.pcolormesh(lon, lat, Aver_SST, cmap=cmap, vmin=20, vmax=23, transform=proj)

cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.set_label(r'[$^\circ$C]', fontsize=20)
ax.coastlines(resolution='10m', color='black', linewidth=1)
ax.add_feature(land_10m)
ax.add_feature(cft.BORDERS)
# ax.set_extent([-22, -11, 24, 33], crs=proj)  #Canarias
ax.set_extent([-8, -5.5, 37.5, 35.5], crs=proj)  #Golfo Cadiz
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks([-22,-20,-18,-16,-14,-12], crs=proj) #Canarias
# ax.set_yticks([24,26,28,30,32], crs=proj)          #Canarias
ax.set_xticks([-8, -7, -6], crs=proj) #Golfo Cadiz
ax.set_yticks([37.5, 37, 36.5, 36, 35.5], crs=proj)          #Golfo Cadiz
#ax.set_extent(img_extent, crs=proj_carr)
#Set the title
plt.title('Averaged SST 2022', fontsize=30)








#Pintar MHWs Estrategias Marinas

                        #################
                        ###  CANARIAS ###
                        #################

#################
# MHW frequency #
#################

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
# levels=np.linspace(np.nanmin(MHW_cnt_canarias), np.nanmax(MHW_cnt_canarias), num=21)
levels=np.linspace(1.5, 2.5, num=21)
cs= ax.contourf(LON_canarias, LAT_canarias, MHW_cnt_canarias, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([1.5, 1.7, 1.9, 2.1, 2.3, 2.5])
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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
plt.title('MHW Frequency [$number$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Freq_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



#######################
# MHW Frequency trend #
#######################

signif_freq_canarias = MHW_cnt_dtr_canarias
signif_freq_canarias = np.where(signif_freq_canarias >= 0.05, np.NaN, signif_freq_canarias)
signif_freq_canarias = np.where(signif_freq_canarias < 0.05, 1, signif_freq_canarias)

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

# levels=np.linspace(np.nanmin(MHW_cnt_tr_canarias*10), np.nanmax(MHW_cnt_tr_canarias*10), num=20)
levels=np.linspace(-1, 1, 21)
cs= ax.contourf(LON_canarias, LAT_canarias, MHW_cnt_tr_canarias*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_canarias[::2,::2], LAT_canarias[::2,::2], signif_freq_canarias[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-1, -0.5, 0, 0.5, 1])

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
plt.title('MHW Frequency trend [$number·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Freq_tr_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



################
# MHW duration #
################

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
levels=np.linspace(12, 22, num=21)


cs= ax.contourf(LON_canarias, LAT_canarias, MHW_dur_canarias, levels, cmap=cmap, transform=proj, extend ='both')

cbar = plt.colorbar(cs, shrink=0.67, extend ='both')
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([12, 14, 16, 18, 20, 22])

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


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Dur_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




#######################
# MHW Duration trend ##
#######################

signif_dur_canarias = MHW_dur_dtr_canarias
signif_dur_canarias = np.where(signif_dur_canarias >= 0.05, np.NaN, signif_dur_canarias)
signif_dur_canarias = np.where(signif_dur_canarias < 0.05, 1, signif_dur_canarias)

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
levels=np.linspace(-6, 6, num=21)

cs= ax.contourf(LON_canarias, LAT_canarias, MHW_dur_tr_canarias*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_canarias[::2,::2], LAT_canarias[::2,::2], signif_dur_canarias[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Dur_tr_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





################
# MHW Mean Int #
################

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
levels=np.linspace(np.nanmin(MHW_mean_canarias), np.nanmax(MHW_mean_canarias), num=21)

cs= ax.contourf(LON_canarias, LAT_canarias, MHW_mean_canarias, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
# cbar.set_label(r'[$days$]', fontsize=20)
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


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MeanInt_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




#######################
# MHW Mean Int trend ##
#######################

signif_mean_canarias = MHW_mean_dtr_canarias
signif_mean_canarias = np.where(signif_mean_canarias >= 0.05, np.NaN, signif_mean_canarias)
signif_mean_canarias = np.where(signif_mean_canarias < 0.05, 1, signif_mean_canarias)

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
levels=np.linspace(-0.1, 0.1, num=21)

cs= ax.contourf(LON_canarias, LAT_canarias, MHW_mean_tr_canarias*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_canarias[::2,::2], LAT_canarias[::2,::2], signif_mean_canarias[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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
plt.title('MHW Mean intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MeanInt_tr_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)


################
# MHW Max Int ##
################

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
# levels=np.linspace(np.nanmin(MHW_max_canarias), np.nanmax(MHW_max_canarias), num=21)
levels=np.linspace(1.3, 2.1, num=21)

cs= ax.contourf(LON_canarias, LAT_canarias, MHW_max_canarias, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1])
# cbar.set_label(r'[$days$]', fontsize=20)
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


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MaxInt_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




#######################
# MHW Max Int trend ###
#######################

signif_max_canarias = MHW_max_dtr_canarias
signif_max_canarias = np.where(signif_max_canarias >= 0.05, np.NaN, signif_max_canarias)
signif_max_canarias = np.where(signif_max_canarias < 0.05, 1, signif_max_canarias)

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
# levels=np.linspace(np.nanmin(MHW_mean_tr_canarias*10), np.nanmax(MHW_mean_tr_canarias*10), num=20)
levels=np.linspace(-0.14, 0.14, num=21)

cs= ax.contourf(LON_canarias, LAT_canarias, MHW_max_tr_canarias*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_canarias[::2,::2], LAT_canarias[::2,::2], signif_max_canarias[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([-0.14, -0.07, 0.00, 0.07, 0.14])
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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
plt.title('MHW Max intensity trend [$^{\circ}C·decade^{-1}$]', fontsize=25)


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_MaxInt_tr_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



##########################
# Total Annual MHW days ##
##########################

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
# levels=np.linspace(np.nanmin(MHW_td_canarias), np.nanmax(MHW_td_canarias), num=21)
levels=np.linspace(25, 35, num=21)

cs= ax.contourf(LON_canarias, LAT_canarias, MHW_td_canarias, levels, cmap=cmap, transform=proj, extend ='both')


cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=18) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([25, 27.5, 30, 32.5, 35])
# cbar.set_label(r'[$days$]', fontsize=20)
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


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Td_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



#################################
# Total Annual MHW days trend ###
#################################

signif_td_canarias = MHW_td_dtr_canarias
signif_td_canarias = np.where(signif_td_canarias >= 0.05, np.NaN, signif_td_canarias)
signif_td_canarias = np.where(signif_td_canarias < 0.05, 1, signif_td_canarias)

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
# levels=np.linspace(np.nanmin(MHW_td_tr_canarias*10), np.nanmax(MHW_td_tr_canarias*10), num=21)
levels=np.linspace(0, 16.5, num=21)

cs= ax.contourf(LON_canarias, LAT_canarias, MHW_td_tr_canarias*10, levels, cmap=cmap, transform=proj, extend ='both')
cs2=ax.scatter(LON_canarias[::2,::2], LAT_canarias[::2,::2], signif_td_canarias[::2,::2], color='black',linewidth=1,marker='o', alpha=0.8)


cbar = plt.colorbar(cs, shrink=0.67, extend ='both', format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=8, direction='in', labelsize=20) 
cbar.ax.minorticks_off()
cbar.ax.get_yaxis().set_ticks([0, 2.5, 5, 7.5, 10, 12.5, 15])
# cbar.set_label(r'Frequency [$number$]', fontsize=20)
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


outfile = r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Figuras_MHWs\MHW_Td_tr_canarias.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



