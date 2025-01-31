# -*- coding: utf-8 -*-
"""

###########################  CHL Data Clipper  ################################

"""

#Loading required libraries
import geopandas as gpd

import xarray as xr 

from shapely.geometry import mapping


##Loading raw data
# ds = xr.open_mfdataset(r'E:\ICMAN-CSIC\CHL_Phenology\CHL_MODIS_L3_4Km_8day_Data\RAW/*.nc', parallel=True)

ds = xr.open_dataset(r'E:\.../CHL_L4_1Km.nc')

##Setting up ds
ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds.rio.write_crs("epsg:4326", inplace=True)

##Loading shapefiles
Shapefile_CAN = gpd.read_file(r'E:\...\cn.shp', crs="epsg:4326")
Shapefile_NA = gpd.read_file(r'E:\...\gl.shp', crs="epsg:4326")
Shapefile_SA = gpd.read_file(r'E:\...\sa.shp', crs="epsg:4326")
Shapefile_AL = gpd.read_file(r'E:\...\alb.shp', crs="epsg:4326")
Shapefile_BAL = gpd.read_file(r'E:\...\bl.shp', crs="epsg:4326")

##Clipping dataset to shapefile dimension
#CAN
clipped = ds.CHL.rio.clip(Shapefile_CAN.geometry.apply(mapping), Shapefile_CAN.crs, drop=True)
clipped.to_netcdf(r'E:\.../CHL_L4_1km_CAN.nc')

#NA
clipped = ds.CHL.rio.clip(Shapefile_NA.geometry.apply(mapping), Shapefile_NA.crs, drop=True)
clipped.to_netcdf(r'E:\.../CHL_L4_1km_NA.nc')

#SA
clipped = ds.CHL.rio.clip(Shapefile_SA.geometry.apply(mapping), Shapefile_SA.crs, drop=True)
clipped.to_netcdf(r'E:\.../CHL_L4_1km_SA.nc')

#AL
clipped = ds.CHL.rio.clip(Shapefile_AL.geometry.apply(mapping), Shapefile_AL.crs, drop=True)
clipped.to_netcdf(r'E:\.../CHL_L4_1km_AL.nc')

#BAL
clipped = ds.CHL.rio.clip(Shapefile_BAL.geometry.apply(mapping), Shapefile_BAL.crs, drop=True)
clipped.to_netcdf(r'E:\.../CHL_L4_1km_BAL.nc')

