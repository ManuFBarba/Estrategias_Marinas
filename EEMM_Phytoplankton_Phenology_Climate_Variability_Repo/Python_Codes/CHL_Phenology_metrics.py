# -*- coding: utf-8 -*-
"""

############### CHL Phenology Metrics and Climate Variables ###################

"""

#Loading required Python modules
import numpy as np
import xarray as xr 


##Loading previously processed phenology metrics (.npy)


## Canary (CAN)
directory = r'E:\...\Phenology_Metrics\CAN/'

#Lat and Lon
LAT_CAN = np.load(directory + 'LAT_CAN.npy')
LON_CAN = np.load(directory + 'LON_CAN.npy')

#Time series metrics
SCR_CAN = np.load(directory + 'SCR_CAN.npy')
bloom_freq_CAN = np.load(directory + 'bloom_freq_CAN.npy')
bloom_max_CAN = np.load(directory + 'bloom_max_CAN.npy')
bloom_peak_CAN = np.load(directory + 'bloom_peak_CAN.npy')
bloom_amp_CAN = np.load(directory + 'bloom_amp_CAN.npy')
bloom_ini_CAN = np.load(directory + 'bloom_ini_CAN.npy')
bloom_fin_CAN = np.load(directory + 'bloom_fin_CAN.npy')
bloom_dur_CAN = np.load(directory + 'bloom_dur_CAN.npy')
bloom_main_cum_CAN = np.load(directory + 'bloom_main_cum_CAN.npy')
bloom_total_cum_CAN = np.load(directory + 'bloom_total_cum_CAN.npy')

#Decadal trends and significance
# SCR_trend_CAN = np.load(directory + 'SCR_trend_CAN.npy')
# SCR_significance_CAN = np.load(directory + 'SCR_significance_CAN.npy')

bloom_freq_trend_CAN = np.load(directory + 'bloom_freq_trend_CAN.npy')
bloom_freq_significance_CAN = np.load(directory + 'bloom_freq_significance_CAN.npy')

bloom_max_trend_CAN = np.load(directory + 'bloom_max_trend_CAN.npy')
bloom_max_significance_CAN = np.load(directory + 'bloom_max_significance_CAN.npy')

bloom_peak_trend_CAN = np.load(directory + 'bloom_peak_trend_CAN.npy')
bloom_peak_significance_CAN = np.load(directory + 'bloom_peak_significance_CAN.npy')

bloom_amp_trend_CAN = np.load(directory + 'bloom_amp_trend_CAN.npy')
bloom_amp_significance_CAN = np.load(directory + 'bloom_amp_significance_CAN.npy')

bloom_ini_trend_CAN = np.load(directory + 'bloom_ini_trend_CAN.npy')
bloom_ini_significance_CAN = np.load(directory + 'bloom_ini_significance_CAN.npy')

bloom_fin_trend_CAN = np.load(directory + 'bloom_fin_trend_CAN.npy')
bloom_fin_significance_CAN = np.load(directory + 'bloom_fin_significance_CAN.npy')

bloom_dur_trend_CAN = np.load(directory + 'bloom_dur_trend_CAN.npy')
bloom_dur_significance_CAN = np.load(directory + 'bloom_dur_significance_CAN.npy')

bloom_main_cum_trend_CAN = np.load(directory + 'bloom_main_cum_trend_CAN.npy')
bloom_main_cum_significance_CAN = np.load(directory + 'bloom_main_cum_significance_CAN.npy')

bloom_total_cum_trend_CAN = np.load(directory + 'bloom_total_cum_trend_CAN.npy')
bloom_total_cum_significance_CAN = np.load(directory + 'bloom_total_cum_significance_CAN.npy')


## North Atlantic (NA)
directory = r'E:\...\Phenology_Metrics\NA/'

#Lat and Lon
LAT_NA = np.load(directory + 'LAT_NA.npy')
LON_NA = np.load(directory + 'LON_NA.npy')

#Time series metrics
SCR_NA = np.load(directory + 'SCR_NA.npy')
bloom_freq_NA = np.load(directory + 'bloom_freq_NA.npy')
bloom_max_NA = np.load(directory + 'bloom_max_NA.npy')
bloom_peak_NA = np.load(directory + 'bloom_peak_NA.npy')
bloom_amp_NA = np.load(directory + 'bloom_amp_NA.npy')
bloom_ini_NA = np.load(directory + 'bloom_ini_NA.npy')
bloom_fin_NA = np.load(directory + 'bloom_fin_NA.npy')
bloom_dur_NA = np.load(directory + 'bloom_dur_NA.npy')
bloom_main_cum_NA = np.load(directory + 'bloom_main_cum_NA.npy')
bloom_total_cum_NA = np.load(directory + 'bloom_total_cum_NA.npy')

#Decadal trends and significance
# SCR_trend_NA = np.load(directory + 'SCR_trend_NA.npy')
# SCR_significance_NA = np.load(directory + 'SCR_significance_NA.npy')

bloom_freq_trend_NA = np.load(directory + 'bloom_freq_trend_NA.npy')
bloom_freq_significance_NA = np.load(directory + 'bloom_freq_significance_NA.npy')

bloom_max_trend_NA = np.load(directory + 'bloom_max_trend_NA.npy')
bloom_max_significance_NA = np.load(directory + 'bloom_max_significance_NA.npy')

bloom_peak_trend_NA = np.load(directory + 'bloom_peak_trend_NA.npy')
bloom_peak_significance_NA = np.load(directory + 'bloom_peak_significance_NA.npy')

bloom_amp_trend_NA = np.load(directory + 'bloom_amp_trend_NA.npy')
bloom_amp_significance_NA = np.load(directory + 'bloom_amp_significance_NA.npy')

bloom_ini_trend_NA = np.load(directory + 'bloom_ini_trend_NA.npy')
bloom_ini_significance_NA = np.load(directory + 'bloom_ini_significance_NA.npy')

bloom_fin_trend_NA = np.load(directory + 'bloom_fin_trend_NA.npy')
bloom_fin_significance_NA = np.load(directory + 'bloom_fin_significance_NA.npy')

bloom_dur_trend_NA = np.load(directory + 'bloom_dur_trend_NA.npy')
bloom_dur_significance_NA = np.load(directory + 'bloom_dur_significance_NA.npy')

bloom_main_cum_trend_NA = np.load(directory + 'bloom_main_cum_trend_NA.npy')
bloom_main_cum_significance_NA = np.load(directory + 'bloom_main_cum_significance_NA.npy')

bloom_total_cum_trend_NA = np.load(directory + 'bloom_total_cum_trend_NA.npy')
bloom_total_cum_significance_NA = np.load(directory + 'bloom_total_cum_significance_NA.npy')


## South Atlantic (SA)
directory = r'E:\...\Phenology_Metrics\SA/'

#Lat and Lon
LAT_SA = np.load(directory + 'LAT_SA.npy')
LON_SA = np.load(directory + 'LON_SA.npy')

#Time series metrics
SCR_SA = np.load(directory + 'SCR_SA.npy')
bloom_freq_SA = np.load(directory + 'bloom_freq_SA.npy')
bloom_max_SA = np.load(directory + 'bloom_max_SA.npy')
bloom_peak_SA = np.load(directory + 'bloom_peak_SA.npy')
bloom_amp_SA = np.load(directory + 'bloom_amp_SA.npy')
bloom_ini_SA = np.load(directory + 'bloom_ini_SA.npy')
bloom_fin_SA = np.load(directory + 'bloom_fin_SA.npy')
bloom_dur_SA = np.load(directory + 'bloom_dur_SA.npy')
bloom_main_cum_SA = np.load(directory + 'bloom_main_cum_SA.npy')
bloom_total_cum_SA = np.load(directory + 'bloom_total_cum_SA.npy')

#Decadal trends and significance
# SCR_trend_SA = np.load(directory + 'SCR_trend_SA.npy')
# SCR_significance_SA = np.load(directory + 'SCR_significance_SA.npy')

bloom_freq_trend_SA = np.load(directory + 'bloom_freq_trend_SA.npy')
bloom_freq_significance_SA = np.load(directory + 'bloom_freq_significance_SA.npy')

bloom_max_trend_SA = np.load(directory + 'bloom_max_trend_SA.npy')
bloom_max_significance_SA = np.load(directory + 'bloom_max_significance_SA.npy')

bloom_peak_trend_SA = np.load(directory + 'bloom_peak_trend_SA.npy')
bloom_peak_significance_SA = np.load(directory + 'bloom_peak_significance_SA.npy')

bloom_amp_trend_SA = np.load(directory + 'bloom_amp_trend_SA.npy')
bloom_amp_significance_SA = np.load(directory + 'bloom_amp_significance_SA.npy')

bloom_ini_trend_SA = np.load(directory + 'bloom_ini_trend_SA.npy')
bloom_ini_significance_SA = np.load(directory + 'bloom_ini_significance_SA.npy')

bloom_fin_trend_SA = np.load(directory + 'bloom_fin_trend_SA.npy')
bloom_fin_significance_SA = np.load(directory + 'bloom_fin_significance_SA.npy')

bloom_dur_trend_SA = np.load(directory + 'bloom_dur_trend_SA.npy')
bloom_dur_significance_SA = np.load(directory + 'bloom_dur_significance_SA.npy')

bloom_main_cum_trend_SA = np.load(directory + 'bloom_main_cum_trend_SA.npy')
bloom_main_cum_significance_SA = np.load(directory + 'bloom_main_cum_significance_SA.npy')

bloom_total_cum_trend_SA = np.load(directory + 'bloom_total_cum_trend_SA.npy')
bloom_total_cum_significance_SA = np.load(directory + 'bloom_total_cum_significance_SA.npy')



## Strait of Gibraltar and Alboran Sea (AL)
directory = r'E:\...\Phenology_Metrics\AL/'

#Lat and Lon
LAT_AL = np.load(directory + 'LAT_AL.npy')
LON_AL = np.load(directory + 'LON_AL.npy')

#Time series metrics
SCR_AL = np.load(directory + 'SCR_AL.npy')
bloom_freq_AL = np.load(directory + 'bloom_freq_AL.npy')
bloom_max_AL = np.load(directory + 'bloom_max_AL.npy')
bloom_peak_AL = np.load(directory + 'bloom_peak_AL.npy')
bloom_amp_AL = np.load(directory + 'bloom_amp_AL.npy')
bloom_ini_AL = np.load(directory + 'bloom_ini_AL.npy')
bloom_fin_AL = np.load(directory + 'bloom_fin_AL.npy')
bloom_dur_AL = np.load(directory + 'bloom_dur_AL.npy')
bloom_main_cum_AL = np.load(directory + 'bloom_main_cum_AL.npy')
bloom_total_cum_AL = np.load(directory + 'bloom_total_cum_AL.npy')

#Decadal trends and significance
# SCR_trend_AL = np.load(directory + 'SCR_trend_AL.npy')
# SCR_significance_AL = np.load(directory + 'SCR_significance_AL.npy')

bloom_freq_trend_AL = np.load(directory + 'bloom_freq_trend_AL.npy')
bloom_freq_significance_AL = np.load(directory + 'bloom_freq_significance_AL.npy')

bloom_max_trend_AL = np.load(directory + 'bloom_max_trend_AL.npy')
bloom_max_significance_AL = np.load(directory + 'bloom_max_significance_AL.npy')

bloom_peak_trend_AL = np.load(directory + 'bloom_peak_trend_AL.npy')
bloom_peak_significance_AL = np.load(directory + 'bloom_peak_significance_AL.npy')

bloom_amp_trend_AL = np.load(directory + 'bloom_amp_trend_AL.npy')
bloom_amp_significance_AL = np.load(directory + 'bloom_amp_significance_AL.npy')

bloom_ini_trend_AL = np.load(directory + 'bloom_ini_trend_AL.npy')
bloom_ini_significance_AL = np.load(directory + 'bloom_ini_significance_AL.npy')

bloom_fin_trend_AL = np.load(directory + 'bloom_fin_trend_AL.npy')
bloom_fin_significance_AL = np.load(directory + 'bloom_fin_significance_AL.npy')

bloom_dur_trend_AL = np.load(directory + 'bloom_dur_trend_AL.npy')
bloom_dur_significance_AL = np.load(directory + 'bloom_dur_significance_AL.npy')

bloom_main_cum_trend_AL = np.load(directory + 'bloom_main_cum_trend_AL.npy')
bloom_main_cum_significance_AL = np.load(directory + 'bloom_main_cum_significance_AL.npy')

bloom_total_cum_trend_AL = np.load(directory + 'bloom_total_cum_trend_AL.npy')
bloom_total_cum_significance_AL = np.load(directory + 'bloom_total_cum_significance_AL.npy')



## Levantine-Balearic (BAL)
directory = r'E:\...\Phenology_Metrics\BAL/'

#Lat and Lon
LAT_BAL = np.load(directory + 'LAT_BAL.npy')
LON_BAL = np.load(directory + 'LON_BAL.npy')

#Time series metrics
SCR_BAL = np.load(directory + 'SCR_BAL.npy')
bloom_freq_BAL = np.load(directory + 'bloom_freq_BAL.npy')
bloom_max_BAL = np.load(directory + 'bloom_max_BAL.npy')
bloom_peak_BAL = np.load(directory + 'bloom_peak_BAL.npy')
bloom_amp_BAL = np.load(directory + 'bloom_amp_BAL.npy')
bloom_ini_BAL = np.load(directory + 'bloom_ini_BAL.npy')
bloom_fin_BAL = np.load(directory + 'bloom_fin_BAL.npy')
bloom_dur_BAL = np.load(directory + 'bloom_dur_BAL.npy')
bloom_main_cum_BAL = np.load(directory + 'bloom_main_cum_BAL.npy')
bloom_total_cum_BAL = np.load(directory + 'bloom_total_cum_BAL.npy')

#Decadal trends and significance
# SCR_trend_BAL = np.load(directory + 'SCR_trend_BAL.npy')
# SCR_significance_BAL = np.load(directory + 'SCR_significance_BAL.npy')

bloom_freq_trend_BAL = np.load(directory + 'bloom_freq_trend_BAL.npy')
bloom_freq_significance_BAL = np.load(directory + 'bloom_freq_significance_BAL.npy')

bloom_max_trend_BAL = np.load(directory + 'bloom_max_trend_BAL.npy')
bloom_max_significance_BAL = np.load(directory + 'bloom_max_significance_BAL.npy')

bloom_peak_trend_BAL = np.load(directory + 'bloom_peak_trend_BAL.npy')
bloom_peak_significance_BAL = np.load(directory + 'bloom_peak_significance_BAL.npy')

bloom_amp_trend_BAL = np.load(directory + 'bloom_amp_trend_BAL.npy')
bloom_amp_significance_BAL = np.load(directory + 'bloom_amp_significance_BAL.npy')

bloom_ini_trend_BAL = np.load(directory + 'bloom_ini_trend_BAL.npy')
bloom_ini_significance_BAL = np.load(directory + 'bloom_ini_significance_BAL.npy')

bloom_fin_trend_BAL = np.load(directory + 'bloom_fin_trend_BAL.npy')
bloom_fin_significance_BAL = np.load(directory + 'bloom_fin_significance_BAL.npy')

bloom_dur_trend_BAL = np.load(directory + 'bloom_dur_trend_BAL.npy')
bloom_dur_significance_BAL = np.load(directory + 'bloom_dur_significance_BAL.npy')

bloom_main_cum_trend_BAL = np.load(directory + 'bloom_main_cum_trend_BAL.npy')
bloom_main_cum_significance_BAL = np.load(directory + 'bloom_main_cum_significance_BAL.npy')

bloom_total_cum_trend_BAL = np.load(directory + 'bloom_total_cum_trend_BAL.npy')
bloom_total_cum_significance_BAL = np.load(directory + 'bloom_total_cum_significance_BAL.npy')



                #####################
                ## Bathymetry Data ##
                #####################

ds_BAT_canarias = xr.open_dataset(r'E:\...\CAN/Bathy_CAN_clipped.nc')
elevation_CAN = ds_BAT_canarias['elevation'] * (-1)
elevation_CAN = xr.where(elevation_CAN <= 0, np.nan, elevation_CAN)
lon_CAN_bat = ds_BAT_canarias['lon']
lat_CAN_bat = ds_BAT_canarias['lat']
LON_CAN_bat, LAT_CAN_bat = np.meshgrid(lon_CAN_bat, lat_CAN_bat)

ds_BAT_SA = xr.open_dataset(r'E:\...\SA/Bathy_SA_clipped.nc')
elevation_SA = ds_BAT_SA['elevation'] * (-1)
elevation_SA = xr.where(elevation_SA <= 0, np.nan, elevation_SA)
lon_SA_bat = ds_BAT_SA['lon']
lat_SA_bat = ds_BAT_SA['lat']
LON_SA_bat, LAT_SA_bat = np.meshgrid(lon_SA_bat, lat_SA_bat)

ds_BAT_AL = xr.open_dataset(r'E:\...\AL/Bathy_AL_clipped.nc')
elevation_AL = ds_BAT_AL['elevation'] * (-1)
elevation_AL = xr.where(elevation_AL <= 0, np.nan, elevation_AL)
lon_AL_bat = ds_BAT_AL['lon']
lat_AL_bat = ds_BAT_AL['lat']
LON_AL_bat, LAT_AL_bat = np.meshgrid(lon_AL_bat, lat_AL_bat)

ds_BAT_BAL = xr.open_dataset(r'E:\...\BAL/Bathy_BAL_clipped.nc')
elevation_BAL = ds_BAT_BAL['elevation'] * (-1)
elevation_BAL = xr.where(elevation_BAL <= 0, np.nan, elevation_BAL)
lon_BAL_bat = ds_BAT_BAL['lon']
lat_BAL_bat = ds_BAT_BAL['lat']
LON_BAL_bat, LAT_BAL_bat = np.meshgrid(lon_BAL_bat, lat_BAL_bat)

ds_BAT_NA = xr.open_dataset(r'E:\...\NA/Bathy_NA_clipped.nc')
elevation_NA = ds_BAT_NA['elevation'] * (-1)
elevation_NA = xr.where(elevation_NA <= 0, np.nan, elevation_NA)
lon_NA_bat = ds_BAT_NA['lon']
lat_NA_bat = ds_BAT_NA['lat']
LON_NA_bat, LAT_NA_bat = np.meshgrid(lon_NA_bat, lat_NA_bat)

