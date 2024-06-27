# -*- coding: utf-8 -*-
"""

#########################      EEMM data processing for 
Fernández-Barba, M., Huertas, I. E., & Navarro, G. (2024). 
Assessment of surface and bottom marine heatwaves along the Spanish coast. 
Ocean Modelling, 190, 102399.                          ########################

"""

#Loading required python modules
import numpy as np
# import pandas as pd
import geopandas as gpd
import xarray as xr 
# import rioxarray

# import rasterio
# import rasterio.mask 

# from shapely.geometry import mapping




################
# Canary (CAN) #
################

##Satellite SST
#Regridding datasets into satellite ds_1 dimensions and Merging 3 datasets in 1
# ds_1 = xr.open_dataset(r'.../SST_1982_1990.nc')
# ds_2 = xr.open_dataset(r'.../SST_1991_2000.nc')
# ds_3 = xr.open_dataset(r'.../SST_2001_2015.nc')
# ds_4 = xr.open_dataset(r'.../SST_2016_2022.nc')
# # ds_4 = ds_2.rename({'sst': 'analysed_sst'})
# #Define the new latitude and longitude grids
# new_lat = np.asarray(ds_4.lat)
# new_lon = np.asarray(ds_4.lon)
# #Interpolate ds_GLORYS, ds_1 and ds_2 datasets to the higher-resolution satellite ds grid
# ds_1 = ds_1.interp(lat=new_lat, lon=new_lon)
# ds_2 = ds_2.interp(lat=new_lat, lon=new_lon)
# ds_3 = ds_3.interp(lat=new_lat, lon=new_lon)
# ds_merged = xr.merge([ds_1, ds_2, ds_3, ds_4])
# #Clipping dataset to shapefile dimension
# ds_merged.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# ds_merged.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file('...\Demarcaciones_Marinas\shapes\canarias\cn.shp', crs="epsg:4326")

# clipped = ds_merged.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../SST_Canary_clipped.nc')

#Load the previously-clipped dataset
ds_canarias = xr.open_dataset(r'.../SST_Canary_clipped.nc')

sst_canarias = ds_canarias['analysed_sst'] - 273.15 #K to ºC
lon_canarias = ds_canarias['lon']
lat_canarias = ds_canarias['lat']

# Select the time period and spatial extent of interest
start_date = '1993-01-01'
end_date = '2022-12-31'
sst_canarias = sst_canarias.sel(time=slice(start_date, end_date))

Aver_SST_canarias =sst_canarias.mean(dim=('time'), skipna=True)#.load() #Average SST 1993-2022
Aver_SST_canarias_ts =sst_canarias.mean(dim=('lon', 'lat'), skipna=True)#.load() #Average SST 1993-2022


##Numerical Model Sea water potential T, bottomT and MLT
#Regridding datasets into satellite ds dimensions and Merging 3 datasets in 1
# ds_GLORYS = xr.open_dataset(r'.../GLORYS_bottomT_T_MLT_1993_2020.nc')
# ds_1 = xr.open_dataset(r'.../T_2021_2022.nc')
# ds_2 = xr.open_dataset(r'.../bottomT_MLT_2021_2022.nc')
# ds_2 = ds_2.rename({'tob': 'bottomT'})
# #Define the new latitude and longitude grids
# new_lat = np.asarray(ds_canarias.lat)
# new_lon = np.asarray(ds_canarias.lon)
# #Interpolate ds_GLORYS, ds_1 and ds_2 datasets to the higher-resolution satellite ds grid
# ds_GLORYS = ds_GLORYS.interp(latitude=new_lat, longitude=new_lon)
# ds_1 = ds_1.interp(latitude=new_lat, longitude=new_lon)
# ds_2 = ds_2.interp(latitude=new_lat, longitude=new_lon)
# ds_merged = xr.merge([ds_GLORYS, ds_1, ds_2])
# #Remove dimension depth
# ds_merged = ds_merged.squeeze('depth')


# #Clipping dataset to shapefile dimension
# ds_merged.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
# ds_merged.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file('...\Demarcaciones_Marinas\shapes\canarias\cn.shp', crs="epsg:4326")
# clipped = ds_merged.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../GLORYS_Canary_clipped.nc')

#Load the previously-clipped dataset
ds_Model_CAN = xr.open_dataset(r'...\Temp_MLT_GLORYS_Canary_clipped.nc')

MLT_CAN = ds_Model_CAN['mlotst']
lon_CAN_MODEL = ds_Model_CAN['longitude']
lat_CAN_MODEL = ds_Model_CAN['latitude']
LON_CAN_MODEL, LAT_CAN_MODEL = np.meshgrid(lon_CAN_MODEL, lat_CAN_MODEL)


thetao_CAN = ds_Model_CAN['thetao']
bottomT_CAN = ds_Model_CAN['bottomT']


##CHL & PFT
#Clipping dataset to shapefile dimension
# OC_dataset = xr.open_mfdataset('...\Canarias\*.nc', parallel=True)
# OC_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# OC_dataset.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file('...\Demarcaciones_Marinas\shapes\canarias\cn.shp', crs="epsg:4326")

# clipped = OC_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../OC_Canarias_clipped.nc')

#Load the previously-clipped OC dataset
ds_OC_canarias = xr.open_dataset(r'.../OC_Canarias_clipped.nc')

CHL_canarias = ds_OC_canarias['CHL']
lon_canarias = ds_OC_canarias['lon']
lat_canarias = ds_OC_canarias['lat']

# Select the time period and spatial extent of interest
start_date = '1998-01-01'
end_date = '2022-12-31'
CHL_canarias = CHL_canarias.sel(time=slice(start_date, end_date))


Aver_CHL_canarias =CHL_canarias.mean(dim=('time'))#.load() #Average CHL 1998-2022
Aver_CHL_canarias.plot()


##BATHYMETRY
#Clipping dataset to shapefile dimension
# BAT_dataset = xr.open_dataset('...\Canarias_Bathymetry.nc')
# BAT_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# BAT_dataset.rio.write_crs("epsg:4326", inplace=True)
Shapefile_Canarias = gpd.read_file('...\Demarcaciones_Marinas\shapes\canarias\cn.shp', crs="epsg:4326")

# clipped = BAT_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped['elevation'].attrs.pop('grid_mapping', None)
# clipped.to_netcdf('.../Bathy_Canarias_clipped.nc')

#Load the previously-clipped Bathymetry dataset
ds_BAT_canarias = xr.open_dataset(r'.../Bathy_Canarias_clipped.nc')

elevation_CAN = ds_BAT_canarias['elevation'] * (-1)
elevation_CAN = xr.where(elevation_CAN <= 0, np.NaN, elevation_CAN)
lon_CAN_bat = ds_BAT_canarias['lon']
lat_CAN_bat = ds_BAT_canarias['lat']
LON_CAN_bat, LAT_CAN_bat = np.meshgrid(lon_CAN_bat, lat_CAN_bat)




#######################
# South Atlantic (SA) #
#######################

##Satellite SST
#Regridding datasets into satellite ds_1 dimensions and Merging 3 datasets in 1
# ds_1 = xr.open_dataset(r'...\Golfo_Cadiz/SST_1982_2020.nc')
# ds_2 = xr.open_dataset(r'...\Golfo_Cadiz/SST_2021_2022.nc')

# # ds_4 = ds_2.rename({'sst': 'analysed_sst'})
# #Define the new latitude and longitude grids
# new_lat = np.asarray(ds_Model_SA.latitude)
# new_lon = np.asarray(ds_Model_SA.longitude)
# #Interpolate ds_GLORYS, ds_1 and ds_2 datasets to the higher-resolution satellite ds grid
# ds_1 = ds_1.interp(lat=new_lat, lon=new_lon)
# ds_2 = ds_2.interp(lat=new_lat, lon=new_lon)

# ds_merged = xr.merge([ds_1, ds_2])

# #Clipping dataset to shapefile dimension
# ds_merged.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# ds_merged.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\sa\sa.shp', crs="epsg:4326")

# clipped = ds_merged.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../SST_SA_clipped.nc')

#Load the previously-clipped dataset
ds_SA = xr.open_dataset(r'.../SST_SA_clipped.nc')

sst_SA = ds_SA['analysed_sst'] - 273.15 #K to ºC
lon_SA = ds_SA['lon']
lat_SA = ds_SA['lat']

# Select the time period and spatial extent of interest
start_date = '1993-01-01'
end_date = '2022-12-31'
sst_SA = sst_SA.sel(time=slice(start_date, end_date))


Aver_SST_SA =sst_SA.mean(dim=('time'), skipna=True)#.load() #Average SST 1993-2022


##Numerical Model Sea water potential T, bottomT and MLT
#Regridding datasets into satellite ds dimensions and Merging 3 datasets in 1
# ds_GLORYS = xr.open_dataset(r'.../GLORYS_bottomT_T_MLT_1993_2020.nc')
# ds_1 = xr.open_dataset(r'.../T_2021_2022.nc')
# ds_2 = xr.open_dataset(r'.../bottomT_MLT_2021_2022.nc')
# ds_2 = ds_2.rename({'tob': 'bottomT'})
# #Define the new latitude and longitude grids
# new_lat = np.asarray(ds_SA.lat)
# new_lon = np.asarray(ds_SA.lon)
# #Interpolate ds_GLORYS, ds_1 and ds_2 datasets to the higher-resolution satellite ds grid
# ds_GLORYS = ds_GLORYS.interp(latitude=new_lat, longitude=new_lon)
# ds_1 = ds_1.interp(latitude=new_lat, longitude=new_lon)
# ds_2 = ds_2.interp(latitude=new_lat, longitude=new_lon)
# ds_merged = xr.merge([ds_GLORYS, ds_1, ds_2])
# #Remove dimension depth
# ds_merged = ds_merged.squeeze('depth')


# #Clipping dataset to shapefile dimension
# ds_merged.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
# ds_merged.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\sa\sa.shp', crs="epsg:4326")
# clipped = ds_merged.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../Temp_GLORYS_SA_clipped.nc')

#Load the previously-clipped dataset
ds_Model_SA = xr.open_dataset(r'...\Temp_MLT_GLORYS_SA_clipped.nc')

MLT_SA = ds_Model_SA['mlotst']
lon_SA_MODEL = ds_Model_SA['longitude']
lat_SA_MODEL = ds_Model_SA['latitude']
LON_SA_MODEL, LAT_SA_MODEL = np.meshgrid(lon_SA_MODEL, lat_SA_MODEL)


thetao_SA = ds_Model_SA['thetao']
bottomT_SA = ds_Model_SA['bottomT']


ds_Model_SA.bottomT.mean(dim='time', skipna=True).plot()


##CHL & PFT
#Clipping dataset to shapefile dimension
# OC_dataset = xr.open_mfdataset('...\Golfo_Cadiz\*.nc', parallel=True)
# OC_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# OC_dataset.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\sa\sa.shp', crs="epsg:4326")

# clipped = OC_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../OC_SA_clipped.nc')

#Load the previously-clipped OC dataset
ds_OC_SA = xr.open_dataset(r'.../OC_SA_clipped.nc')

CHL_SA = ds_OC_SA['CHL']
lon_SA = ds_OC_SA['lon']
lat_SA = ds_OC_SA['lat']

# Select the time period and spatial extent of interest
start_date = '1998-01-01'
end_date = '2022-12-31'
CHL_SA = CHL_SA.sel(time=slice(start_date, end_date))

Aver_CHL_SA =CHL_SA.mean(dim=('time'))#.load() #Average CHL 1998-2022
Aver_CHL_SA.plot()


##BATHYMETRY
#Clipping dataset to shapefile dimension
# BAT_dataset = xr.open_dataset('...\GC_Bathymetry.nc')
# BAT_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# BAT_dataset.rio.write_crs("epsg:4326", inplace=True)
Shapefile_SA = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\sa\sa.shp', crs="epsg:4326")

# clipped = BAT_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped['elevation'].attrs.pop('grid_mapping', None)
# clipped.to_netcdf('.../Bathy_SA_clipped.nc')

#Load the previously-clipped Bathymetry dataset
ds_BAT_SA = xr.open_dataset(r'.../Bathy_SA_clipped.nc')

elevation_SA = ds_BAT_SA['elevation'] * (-1)
elevation_SA = xr.where(elevation_SA <= 0, np.NaN, elevation_SA)
lon_SA_bat = ds_BAT_SA['lon']
lat_SA_bat = ds_BAT_SA['lat']
LON_SA_bat, LAT_SA_bat = np.meshgrid(lon_SA_bat, lat_SA_bat)




############################
# SoG and Alboran Sea (AL) #
############################

##Satellite SST
#Regridding datasets into satellite ds_1 dimensions and Merging 3 datasets in 1
# ds_1 = xr.open_dataset(r'...\Alboran/SST_1982_2020.nc')
# ds_2 = xr.open_dataset(r'...\Alboran/SST_2021_2022.nc')

# # ds_4 = ds_2.rename({'sst': 'analysed_sst'})
# #Define the new latitude and longitude grids
# new_lat = np.asarray(ds_2.lat)
# new_lon = np.asarray(ds_2.lon)
# #Interpolate ds_GLORYS, ds_1 and ds_2 datasets to the higher-resolution satellite ds grid
# ds_1 = ds_1.interp(lat=new_lat, lon=new_lon)

# ds_merged = xr.merge([ds_1, ds_2])

# #Clipping dataset to shapefile dimension
# ds_merged.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# ds_merged.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\alb\alb.shp', crs="epsg:4326")

# clipped = ds_merged.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../SST_AL_clipped.nc')

#Load the previously-clipped dataset
ds_AL = xr.open_dataset(r'.../SST_AL_clipped.nc')

sst_AL = ds_AL['analysed_sst'] - 273.15 #K to ºC
lon_AL = ds_AL['lon']
lat_AL = ds_AL['lat']

# Select the time period and spatial extent of interest
start_date = '1993-01-01'
end_date = '2022-12-31'
sst_AL = sst_AL.sel(time=slice(start_date, end_date))


Aver_SST_AL =sst_AL.mean(dim=('time'), skipna=True)#.load() #Average SST 1993-2022


##Numerical Model Sea water potential T, bottomT and MLT
#Regridding datasets into satellite ds dimensions and Merging 3 datasets in 1
# ds_GLORYS = xr.open_dataset(r'.../GLORYS_bottomT_T_MLT_1993_2020.nc')
# ds_1 = xr.open_dataset(r'.../T_2021_2022.nc')
# ds_2 = xr.open_dataset(r'.../bottomT_MLT_2021_2022.nc')
# ds_2 = ds_2.rename({'tob': 'bottomT'})
# #Define the new latitude and longitude grids
# new_lat = np.asarray(ds_AL.lat)
# new_lon = np.asarray(ds_AL.lon)
# #Interpolate ds_GLORYS, ds_1 and ds_2 datasets to the higher-resolution satellite ds grid
# ds_GLORYS = ds_GLORYS.interp(latitude=new_lat, longitude=new_lon)
# ds_1 = ds_1.interp(latitude=new_lat, longitude=new_lon)
# ds_2 = ds_2.interp(latitude=new_lat, longitude=new_lon)
# ds_merged = xr.merge([ds_GLORYS, ds_1, ds_2])
# #Remove dimension depth
# ds_merged = ds_merged.squeeze('depth')


# #Clipping dataset to shapefile dimension
# ds_merged.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
# ds_merged.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\alb\alb.shp', crs="epsg:4326")
# clipped = ds_merged.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../Temp_GLORYS_AL_clipped.nc')

#Load the previously-clipped dataset
ds_Model_AL = xr.open_dataset(r'...\Temp_MLT_GLORYS_AL_clipped.nc')

MLT_AL = ds_Model_AL['mlotst']
lon_AL_MODEL = ds_Model_AL['longitude']
lat_AL_MODEL = ds_Model_AL['latitude']
LON_AL_MODEL, LAT_AL_MODEL = np.meshgrid(lon_AL_MODEL, lat_AL_MODEL)


thetao_AL = ds_Model_AL['thetao']
bottomT_AL = ds_Model_AL['bottomT']

ds_Model_AL.thetao.mean(dim='time', skipna=True).plot()


##CHL & PFT
#Clipping dataset to shapefile dimension
# OC_dataset = xr.open_mfdataset('...\Alboran\*.nc', parallel=True)
# OC_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# OC_dataset.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\alb\alb.shp', crs="epsg:4326")

# clipped = OC_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../OC_AL_clipped.nc')

#Load the previously-clipped OC dataset
ds_OC_AL = xr.open_dataset(r'.../OC_AL_clipped.nc')

CHL_AL = ds_OC_AL['CHL']
lon_AL = ds_OC_AL['lon']
lat_AL = ds_OC_AL['lat']

# Select the time period and spatial extent of interest
start_date = '1998-01-01'
end_date = '2022-12-31'
CHL_AL = CHL_AL.sel(time=slice(start_date, end_date))

Aver_CHL_AL =CHL_AL.mean(dim=('time'))#.load() #Average CHL 1998-2022
Aver_CHL_AL.plot()


##BATHYMETRY
#Clipping dataset to shapefile dimension
# BAT_dataset = xr.open_dataset('...\AL_Bathymetry.nc')
# BAT_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# BAT_dataset.rio.write_crs("epsg:4326", inplace=True)
Shapefile_AL = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\alb\alb.shp', crs="epsg:4326")

# clipped = BAT_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped['elevation'].attrs.pop('grid_mapping', None)
# clipped.to_netcdf('.../Bathy_AL_clipped.nc')

#Load the previously-clipped Bathymetry dataset
ds_BAT_AL = xr.open_dataset(r'.../Bathy_AL_clipped.nc')

elevation_AL = ds_BAT_AL['elevation'] * (-1)
elevation_AL = xr.where(elevation_AL <= 0, np.NaN, elevation_AL)
lon_AL_bat = ds_BAT_AL['lon']
lat_AL_bat = ds_BAT_AL['lat']
LON_AL_bat, LAT_AL_bat = np.meshgrid(lon_AL_bat, lat_AL_bat)




############################
# Levantine-Balearic (BAL) #
############################

##Satellite SST
#Regridding datasets into satellite ds_1 dimensions and Merging 3 datasets in 1
# ds_1 = xr.open_dataset(r'...\Baleares/SST_1982_2020.nc')
# ds_2 = xr.open_dataset(r'...\Baleares/SST_2021_2022.nc')

# # ds_4 = ds_2.rename({'sst': 'analysed_sst'})
# #Define the new latitude and longitude grids
# new_lat = np.asarray(ds_2.lat)
# new_lon = np.asarray(ds_2.lon)
# #Interpolate ds_GLORYS, ds_1 and ds_2 datasets to the higher-resolution satellite ds grid
# ds_1 = ds_1.interp(lat=new_lat, lon=new_lon)

# ds_merged = xr.merge([ds_1, ds_2])

# #Clipping dataset to shapefile dimension
# ds_merged.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# ds_merged.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\bl\bl.shp', crs="epsg:4326")

# clipped = ds_merged.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../SST_BAL_clipped.nc')

#Load the previously-clipped dataset
ds_BAL = xr.open_dataset(r'.../SST_BAL_clipped.nc')

sst_BAL = ds_BAL['analysed_sst'] - 273.15 #K to ºC
lon_BAL = ds_BAL['lon']
lat_BAL = ds_BAL['lat']

# Select the time period and spatial extent of interest
start_date = '1993-01-01'
end_date = '2022-12-31'
sst_BAL = sst_BAL.sel(time=slice(start_date, end_date))


Aver_SST_BAL =sst_BAL.mean(dim=('time'), skipna=True)#.load() #Average SST 1993-2022


##Numerical Model Sea water potential T, bottomT and MLT
#Regridding datasets into satellite ds dimensions and Merging 3 datasets in 1
# ds_1 = xr.open_dataset(r'...\BAL/T_1993_2005.nc')
# ds_2 = xr.open_dataset(r'...\BAL/T_2006_2015.nc')
# ds_3 = xr.open_dataset(r'...\BAL/T_2016_2020.nc')
# ds_4 = xr.open_dataset(r'...\BAL/T_2021_2022.nc')
# # ds_3 = ds_3.rename({'tob': 'bottomT'})
# #Define the new latitude and longitude grids
# new_lat = np.asarray(ds_BAL.lat)
# new_lon = np.asarray(ds_BAL.lon)
# #Interpolate ds_GLORYS, ds_1 and ds_2 datasets to the higher-resolution satellite ds grid
# ds_1 = ds_1.interp(latitude=new_lat, longitude=new_lon)
# ds_2 = ds_2.interp(latitude=new_lat, longitude=new_lon)
# ds_3 = ds_3.interp(latitude=new_lat, longitude=new_lon)
# ds_4 = ds_4.interp(latitude=new_lat, longitude=new_lon)
# ds_merged = xr.merge([ds_1, ds_2, ds_3, ds_4])
# #Remove dimension depth
# ds_merged = ds_merged.squeeze('depth')


# #Clipping dataset to shapefile dimension
# ds_merged.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
# ds_merged.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\bl\bl.shp', crs="epsg:4326")
# clipped = ds_merged.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../T_GLORYS_BAL_clipped.nc')

#Load the previously-clipped dataset
ds_Model_BAL = xr.open_dataset(r'...\MLT_GLORYS_BAL_clipped.nc')

MLT_BAL = ds_Model_BAL['mlotst']
lon_BAL_MODEL = ds_Model_BAL['longitude']
lat_BAL_MODEL = ds_Model_BAL['latitude']
LON_BAL_MODEL, LAT_BAL_MODEL = np.meshgrid(lon_BAL_MODEL, lat_BAL_MODEL)

ds_Model_BAL = xr.open_dataset(r'...\T_GLORYS_BAL_clipped.nc')
thetao_BAL = ds_Model_BAL['thetao']
ds_Model_BAL = xr.open_dataset(r'...\bottomT_GLORYS_BAL_clipped.nc')
bottomT_BAL = ds_Model_BAL['bottomT']

ds_Model_BAL.thetao.mean(dim='time', skipna=True).plot(cmap='magma_r')


##CHL & PFT
#Clipping dataset to shapefile dimension
# OC_dataset = xr.open_mfdataset('...\Baleares\*.nc', parallel=True)
# OC_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# OC_dataset.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\bl\bl.shp', crs="epsg:4326")

# clipped = OC_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../OC_BAL_clipped.nc')

#Load the previously-clipped OC dataset
ds_OC_BAL = xr.open_dataset(r'.../OC_BAL_clipped.nc')

CHL_BAL = ds_OC_BAL['CHL']
lon_BAL = ds_OC_BAL['lon']
lat_BAL = ds_OC_BAL['lat']

# Select the time period and spatial extent of interest
start_date = '1998-01-01'
end_date = '2022-12-31'
CHL_BAL = CHL_BAL.sel(time=slice(start_date, end_date))

Aver_CHL_BAL =CHL_BAL.mean(dim=('time'))#.load() #Average CHL 1998-2022
Aver_CHL_BAL.plot()


##BATHYMETRY
#Clipping dataset to shapefile dimension
# BAT_dataset = xr.open_dataset('.../BAL_Bathymetry.nc')
# BAT_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# BAT_dataset.rio.write_crs("epsg:4326", inplace=True)
Shapefile_BAL = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\bl\bl.shp', crs="epsg:4326")

# clipped = BAT_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped['elevation'].attrs.pop('grid_mapping', None)
# clipped.to_netcdf('.../Bathy_BAL_clipped.nc')

#Load the previously-clipped Bathymetry dataset
ds_BAT_BAL = xr.open_dataset(r'.../Bathy_BAL_clipped.nc')

elevation_BAL = ds_BAT_BAL['elevation'] * (-1)
elevation_BAL = xr.where(elevation_BAL <= 0, np.NaN, elevation_BAL)
lon_BAL_bat = ds_BAT_BAL['lon']
lat_BAL_bat = ds_BAT_BAL['lat']
LON_BAL_bat, LAT_BAL_bat = np.meshgrid(lon_BAL_bat, lat_BAL_bat)




#######################
# North Atlantic (NA) #
#######################

##SST

# #Reading 2 datasets in different grids
# ds_005 = xr.open_dataset(r'.../cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE_1680083317616_1982_2018.nc')
# ds_002 = xr.open_dataset(r'.../IFREMER-ATL-SST-L4-NRT-OBS_FULL_TIME_SERIE_1680083467094_2019_2022.nc')

# # Define the new latitude and longitude grids
# new_lat = ds_002['lat'][:]
# new_lon = ds_002['lon'][:]
# # # Interpolate each dataset to the high-resolution grid
# ds_005_interp = ds_005.interp(lat=new_lat, lon=new_lon)

# # Resample dataset to the desired grid using nearest-neighbor interpolation
# ds_005_resampled = ds_005_interp.reindex(lat=new_lat, lon=new_lon, method='nearest')

# # #Merging both ds in 1 of 0.01º
# ds_merged = xr.merge([ds_005_resampled, ds_002])

# # #Saving new ds so far
# ds_merged.to_netcdf(r'.../SST_NA_merged.nc')

# ##Clipping dataset to shapefile dimension
# SST_dataset = xr.open_dataset(r'.../SST_NA_merged.nc')
# SST_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# SST_dataset.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\galicia\gl.shp', crs="epsg:4326")

# clipped = SST_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf(r'.../SST_NA_clipped.nc')


#Load the previously-clipped dataset
ds_NA = xr.open_dataset(r'.../SST_NA_clipped.nc')

sst_NA = ds_NA['analysed_sst'] - 273.15 #K to ºC
lon_NA = ds_NA['lon']
lat_NA = ds_NA['lat']

# Select the time period and spatial extent of interest
start_date = '1993-01-01'
end_date = '2022-12-31'
sst_NA = sst_NA.sel(time=slice(start_date, end_date))


Aver_SST_NA =sst_NA.mean(dim='time', skipna=True)#.load() #Average SST 1993-2022


##Numerical Model Sea water potential T, bottomT and MLT
#Regridding datasets into satellite ds dimensions and Merging 3 datasets in 1
# ds_1 = xr.open_dataset(r'...\NA/MLT_1993_2015.nc')
# ds_2 = xr.open_dataset(r'...\NA/MLT_2016_2020.nc')
# ds_3 = xr.open_dataset(r'...\NA/MLT_2021_2022.nc')

# # ds_3 = ds_3.rename({'tob': 'bottomT'})
# #Define the new latitude and longitude grids
# new_lat = np.asarray(ds_NA.lat)
# new_lon = np.asarray(ds_NA.lon)
# #Interpolate ds_GLORYS, ds_1 and ds_2 datasets to the higher-resolution satellite ds grid
# ds_1 = ds_1.interp(latitude=new_lat, longitude=new_lon)
# ds_2 = ds_2.interp(latitude=new_lat, longitude=new_lon)
# ds_3 = ds_3.interp(latitude=new_lat, longitude=new_lon)
# ds_merged = xr.merge([ds_1, ds_2, ds_3])
# #Remove dimension depth
# # ds_merged = ds_merged.squeeze('depth')


# #Clipping dataset to shapefile dimension
# ds_merged.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
# ds_merged.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\galicia\gl.shp', crs="epsg:4326")
# clipped = ds_merged.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../MLT_GLORYS_NA_clipped.nc')

#Load the previously-clipped dataset
ds_Model_NA = xr.open_dataset(r'...\MLT_GLORYS_NA_clipped.nc')

MLT_NA = ds_Model_NA['mlotst']
lon_NA_MODEL = ds_Model_NA['longitude']
lat_NA_MODEL = ds_Model_NA['latitude']
LON_NA_MODEL, LAT_NA_MODEL = np.meshgrid(lon_NA_MODEL, lat_NA_MODEL)

ds_Model_NA = xr.open_dataset(r'...\T_GLORYS_NA_clipped.nc')
thetao_NA = ds_Model_NA['thetao']
ds_Model_NA = xr.open_dataset(r'...\bottomT_GLORYS_NA_clipped.nc')
bottomT_NA = ds_Model_NA['bottomT']

# ds_Model_NA.bottomT.mean(dim='time', skipna=True).plot(cmap='magma_r')
ds_Model_NA.thetao.mean(dim='time', skipna=True).plot(cmap='magma_r')
ds_Model_NA.mlotst.mean(dim='time', skipna=True).plot(cmap='magma_r')


##CHL & PFT
#Clipping dataset to shapefile dimension
# OC_dataset = xr.open_mfdataset('.../Noratlantica/*.nc', parallel=True)
# OC_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# OC_dataset.rio.write_crs("epsg:4326", inplace=True)
# Shapefile = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\galicia\gl.shp', crs="epsg:4326")

# clipped = OC_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped.to_netcdf('.../OC_NA_clipped.nc')

#Load the previously-clipped OC dataset
ds_OC_NA = xr.open_dataset(r'.../OC_NA_clipped.nc')

CHL_NA = ds_OC_NA['CHL']
lon_NA = ds_OC_NA['lon']
lat_NA = ds_OC_NA['lat']

# Select the time period and spatial extent of interest
start_date = '1998-01-01'
end_date = '2022-12-31'
CHL_NA = CHL_NA.sel(time=slice(start_date, end_date))

Aver_CHL_NA =CHL_NA.mean(dim=('time'))#.load() #Average CHL 1998-2022
Aver_CHL_NA.plot()


##BATHYMETRY
#Clipping dataset to shapefile dimension
# BAT_dataset = xr.open_dataset('.../NA_Bathymetry.nc')
# BAT_dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# BAT_dataset.rio.write_crs("epsg:4326", inplace=True)
Shapefile_NA = gpd.read_file(r'...\Demarcaciones_Marinas\shapes\galicia\gl.shp', crs="epsg:4326")

# clipped = BAT_dataset.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)
# clipped['elevation'].attrs.pop('grid_mapping', None)
# clipped.to_netcdf('.../Bathy_NA_clipped.nc')

#Load the previously-clipped Bathymetry dataset
ds_BAT_NA = xr.open_dataset(r'.../Bathy_NA_clipped.nc')

elevation_NA = ds_BAT_NA['elevation'] * (-1)
elevation_NA = xr.where(elevation_NA <= 0, np.NaN, elevation_NA)
lon_NA_bat = ds_BAT_NA['lon']
lat_NA_bat = ds_BAT_NA['lat']
LON_NA_bat, LAT_NA_bat = np.meshgrid(lon_NA_bat, lat_NA_bat)
