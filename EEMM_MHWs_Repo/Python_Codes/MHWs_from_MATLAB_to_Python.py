# -*- coding: utf-8 -*-
"""

#########################      .mat to .npy processing for 
Fernández-Barba, M., Huertas, I. E., & Navarro, G. (2024). 
Assessment of surface and bottom marine heatwaves along the Spanish coast. 
Ocean Modelling, 190, 102399.                          ########################

"""

## Load required Python modules
from scipy.io import loadmat
import numpy as np
import xarray as xr 

                                #############
                                # SATELLITE #
                                # 1982-2012 #
                                #############
                                
##################
## Canary (CAN) ##
##################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_CAN_SAT_1 = loadmat(r'...\total_Canary (CAN)_1.mat')

#lat and lon
lat_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['latitude']
lon_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['longitude']
LAT_CAN_SAT_1, LON_CAN_SAT_1 = np.meshgrid(lat_CAN_SAT_1, lon_CAN_SAT_1)


##MHW Duration 
MHW_dur_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_dur']
MHW_dur_tr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_dur_tr']
MHW_dur_dtr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_dur_dtr']
MHW_dur_ts_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_dur_ts']


##MHW Frequency
MHW_cnt_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_cnt']
MHW_cnt_tr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_cnt_tr']
MHW_cnt_dtr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_cnt_dtr']
MHW_cnt_ts_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_cnt_ts']

##MHW Max Int
MHW_max_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_max']
MHW_max_tr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_max_tr']
MHW_max_dtr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_max_dtr']
MHW_max_ts_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_max_ts']


##MHW Mean Int
MHW_mean_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_mean']
MHW_mean_tr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_mean_tr']
MHW_mean_dtr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_mean_dtr']
MHW_mean_ts_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_mean_ts']


##MHW Cum Int
MHW_cum_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_cum']
MHW_cum_tr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_cum_tr']
MHW_cum_dtr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_cum_dtr']
MHW_cum_ts_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_td']
MHW_td_tr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_td_tr']
MHW_td_dtr_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_td_dtr']
MHW_td_ts_CAN_SAT_1 = TOTAL_MHWs_CAN_SAT_1['MHW_td_ts']




#########################
## South Atlantic (SA) ##
#########################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_SA_SAT_1 = loadmat(r'...\total_SA_1.mat')

#lat and lon
lat_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['latitude']
lon_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['longitude']
LAT_SA_SAT_1, LON_SA_SAT_1 = np.meshgrid(lat_SA_SAT_1, lon_SA_SAT_1)


##MHW Duration 
MHW_dur_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_dur']
MHW_dur_tr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_dur_tr']
MHW_dur_dtr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_dur_dtr']
MHW_dur_ts_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_dur_ts']


##MHW Frequency
MHW_cnt_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_cnt']
MHW_cnt_tr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_cnt_tr']
MHW_cnt_dtr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_cnt_dtr']
MHW_cnt_ts_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_cnt_ts']

##MHW Max Int
MHW_max_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_max']
MHW_max_tr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_max_tr']
MHW_max_dtr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_max_dtr']
MHW_max_ts_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_max_ts']


##MHW Mean Int
MHW_mean_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_mean']
MHW_mean_tr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_mean_tr']
MHW_mean_dtr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_mean_dtr']
MHW_mean_ts_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_mean_ts']


##MHW Cum Int
MHW_cum_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_cum']
MHW_cum_tr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_cum_tr']
MHW_cum_dtr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_cum_dtr']
MHW_cum_ts_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_td']
MHW_td_tr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_td_tr']
MHW_td_dtr_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_td_dtr']
MHW_td_ts_SA_SAT_1 = TOTAL_MHWs_SA_SAT_1['MHW_td_ts']




##############################
## SoG and Alboran Sea (AL) ##
##############################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_AL_SAT_1 = loadmat(r'...\total_AL_1.mat')

#lat and lon
lat_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['latitude']
lon_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['longitude']
LAT_AL_SAT_1, LON_AL_SAT_1 = np.meshgrid(lat_AL_SAT_1, lon_AL_SAT_1)


##MHW Duration 
MHW_dur_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_dur']
MHW_dur_tr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_dur_tr']
MHW_dur_dtr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_dur_dtr']
MHW_dur_ts_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_dur_ts']


##MHW Frequency
MHW_cnt_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_cnt']
MHW_cnt_tr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_cnt_tr']
MHW_cnt_dtr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_cnt_dtr']
MHW_cnt_ts_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_cnt_ts']


##MHW Max Int
MHW_max_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_max']
MHW_max_tr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_max_tr']
MHW_max_dtr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_max_dtr']
MHW_max_ts_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_max_ts']


##MHW Mean Int
MHW_mean_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_mean']
MHW_mean_tr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_mean_tr']
MHW_mean_dtr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_mean_dtr']
MHW_mean_ts_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_mean_ts']


##MHW Cum Int
MHW_cum_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_cum']
MHW_cum_tr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_cum_tr']
MHW_cum_dtr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_cum_dtr']
MHW_cum_ts_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_td']
MHW_td_tr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_td_tr']
MHW_td_dtr_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_td_dtr']
MHW_td_ts_AL_SAT_1 = TOTAL_MHWs_AL_SAT_1['MHW_td_ts']




##############################
## Levantine-Balearic (BAL) ##
##############################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_BAL_SAT_1 = loadmat(r'...\total_BAL_1.mat')

#lat and lon
lat_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['latitude']
lon_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['longitude']
LAT_BAL_SAT_1, LON_BAL_SAT_1 = np.meshgrid(lat_BAL_SAT_1, lon_BAL_SAT_1)


##MHW Duration 
MHW_dur_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_dur']
MHW_dur_tr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_dur_tr']
MHW_dur_dtr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_dur_dtr']
MHW_dur_ts_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_dur_ts']


##MHW Frequency
MHW_cnt_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_cnt']
MHW_cnt_tr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_cnt_tr']
MHW_cnt_dtr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_cnt_dtr']
MHW_cnt_ts_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_cnt_ts']


##MHW Max Int
MHW_max_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_max']
MHW_max_tr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_max_tr']
MHW_max_dtr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_max_dtr']
MHW_max_ts_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_max_ts']


##MHW Mean Int
MHW_mean_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_mean']
MHW_mean_tr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_mean_tr']
MHW_mean_dtr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_mean_dtr']
MHW_mean_ts_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_mean_ts']


##MHW Cum Int
MHW_cum_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_cum']
MHW_cum_tr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_cum_tr']
MHW_cum_dtr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_cum_dtr']
MHW_cum_ts_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_td']
MHW_td_tr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_td_tr']
MHW_td_dtr_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_td_dtr']
MHW_td_ts_BAL_SAT_1 = TOTAL_MHWs_BAL_SAT_1['MHW_td_ts']




#########################
## North Atlantic (NA) ##
#########################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_NA_SAT_1 = loadmat(r'...\total_NA_1.mat')

#lat and lon
lat_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['latitude']
lon_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['longitude']
LAT_NA_SAT_1, LON_NA_SAT_1 = np.meshgrid(lat_NA_SAT_1, lon_NA_SAT_1)


##MHW Duration 
MHW_dur_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_dur']
MHW_dur_tr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_dur_tr']
MHW_dur_dtr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_dur_dtr']
MHW_dur_ts_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_dur_ts']


##MHW Frequency
MHW_cnt_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_cnt']
MHW_cnt_tr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_cnt_tr']
MHW_cnt_dtr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_cnt_dtr']
MHW_cnt_ts_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_cnt_ts']


##MHW Max Int
MHW_max_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_max']
MHW_max_tr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_max_tr']
MHW_max_dtr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_max_dtr']
MHW_max_ts_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_max_ts']


##MHW Mean Int
MHW_mean_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_mean']
MHW_mean_tr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_mean_tr']
MHW_mean_dtr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_mean_dtr']
MHW_mean_ts_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_mean_ts']


##MHW Cum Int
MHW_cum_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_cum']
MHW_cum_tr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_cum_tr']
MHW_cum_dtr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_cum_dtr']
MHW_cum_ts_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_td']
MHW_td_tr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_td_tr']
MHW_td_dtr_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_td_dtr']
MHW_td_ts_NA_SAT_1 = TOTAL_MHWs_NA_SAT_1['MHW_td_ts']




                                #############
                                # SATELLITE #
                                # 1993-2022 #
                                #############
                                
##################
## Canary (CAN) ##
##################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_CAN_SAT = loadmat(r'...\total_CAN_SAT.mat')

#lat and lon
lat_CAN_SAT = TOTAL_MHWs_CAN_SAT['latitude']
lon_CAN_SAT = TOTAL_MHWs_CAN_SAT['longitude']
LAT_CAN_SAT, LON_CAN_SAT = np.meshgrid(lat_CAN_SAT, lon_CAN_SAT)


##MHW Duration 
MHW_dur_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_dur']
MHW_dur_tr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_dur_tr']
MHW_dur_dtr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_dur_dtr']
MHW_dur_ts_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_dur_ts']


##MHW Frequency
MHW_cnt_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_cnt']
MHW_cnt_tr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_cnt_tr']
MHW_cnt_dtr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_cnt_dtr']
MHW_cnt_ts_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_cnt_ts']

##MHW Max Int
MHW_max_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_max']
MHW_max_tr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_max_tr']
MHW_max_dtr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_max_dtr']
MHW_max_ts_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_max_ts']


##MHW Mean Int
MHW_mean_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_mean']
MHW_mean_tr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_mean_tr']
MHW_mean_dtr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_mean_dtr']
MHW_mean_ts_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_mean_ts']


##MHW Cum Int
MHW_cum_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_cum']
MHW_cum_tr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_cum_tr']
MHW_cum_dtr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_cum_dtr']
MHW_cum_ts_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_td']
MHW_td_tr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_td_tr']
MHW_td_dtr_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_td_dtr']
MHW_td_ts_CAN_SAT = TOTAL_MHWs_CAN_SAT['MHW_td_ts']




#########################
## South Atlantic (SA) ##
#########################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_SA_SAT = loadmat(r'...\total_SA_SAT.mat')

#lat and lon
lat_SA_SAT = TOTAL_MHWs_SA_SAT['latitude']
lon_SA_SAT = TOTAL_MHWs_SA_SAT['longitude']
LAT_SA_SAT, LON_SA_SAT = np.meshgrid(lat_SA_SAT, lon_SA_SAT)


##MHW Duration 
MHW_dur_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_dur']
MHW_dur_tr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_dur_tr']
MHW_dur_dtr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_dur_dtr']
MHW_dur_ts_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_dur_ts']


##MHW Frequency
MHW_cnt_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_cnt']
MHW_cnt_tr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_cnt_tr']
MHW_cnt_dtr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_cnt_dtr']
MHW_cnt_ts_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_cnt_ts']

##MHW Max Int
MHW_max_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_max']
MHW_max_tr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_max_tr']
MHW_max_dtr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_max_dtr']
MHW_max_ts_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_max_ts']


##MHW Mean Int
MHW_mean_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_mean']
MHW_mean_tr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_mean_tr']
MHW_mean_dtr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_mean_dtr']
MHW_mean_ts_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_mean_ts']


##MHW Cum Int
MHW_cum_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_cum']
MHW_cum_tr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_cum_tr']
MHW_cum_dtr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_cum_dtr']
MHW_cum_ts_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_td']
MHW_td_tr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_td_tr']
MHW_td_dtr_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_td_dtr']
MHW_td_ts_SA_SAT = TOTAL_MHWs_SA_SAT['MHW_td_ts']




##############################
## SoG and Alboran Sea (AL) ##
##############################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_AL_SAT = loadmat(r'...\total_AL_SAT.mat')

#lat and lon
lat_AL_SAT = TOTAL_MHWs_AL_SAT['latitude']
lon_AL_SAT = TOTAL_MHWs_AL_SAT['longitude']
LAT_AL_SAT, LON_AL_SAT = np.meshgrid(lat_AL_SAT, lon_AL_SAT)


##MHW Duration 
MHW_dur_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_dur']
MHW_dur_tr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_dur_tr']
MHW_dur_dtr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_dur_dtr']
MHW_dur_ts_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_dur_ts']


##MHW Frequency
MHW_cnt_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_cnt']
MHW_cnt_tr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_cnt_tr']
MHW_cnt_dtr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_cnt_dtr']
MHW_cnt_ts_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_cnt_ts']


##MHW Max Int
MHW_max_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_max']
MHW_max_tr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_max_tr']
MHW_max_dtr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_max_dtr']
MHW_max_ts_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_max_ts']


##MHW Mean Int
MHW_mean_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_mean']
MHW_mean_tr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_mean_tr']
MHW_mean_dtr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_mean_dtr']
MHW_mean_ts_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_mean_ts']


##MHW Cum Int
MHW_cum_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_cum']
MHW_cum_tr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_cum_tr']
MHW_cum_dtr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_cum_dtr']
MHW_cum_ts_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_td']
MHW_td_tr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_td_tr']
MHW_td_dtr_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_td_dtr']
MHW_td_ts_AL_SAT = TOTAL_MHWs_AL_SAT['MHW_td_ts']




##############################
## Levantine-Balearic (BAL) ##
##############################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_BAL_SAT = loadmat(r'...\total_BAL_SAT.mat')

#lat and lon
lat_BAL_SAT = TOTAL_MHWs_BAL_SAT['latitude']
lon_BAL_SAT = TOTAL_MHWs_BAL_SAT['longitude']
LAT_BAL_SAT, LON_BAL_SAT = np.meshgrid(lat_BAL_SAT, lon_BAL_SAT)


##MHW Duration 
MHW_dur_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_dur']
MHW_dur_tr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_dur_tr']
MHW_dur_dtr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_dur_dtr']
MHW_dur_ts_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_dur_ts']


##MHW Frequency
MHW_cnt_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_cnt']
MHW_cnt_tr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_cnt_tr']
MHW_cnt_dtr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_cnt_dtr']
MHW_cnt_ts_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_cnt_ts']


##MHW Max Int
MHW_max_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_max']
MHW_max_tr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_max_tr']
MHW_max_dtr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_max_dtr']
MHW_max_ts_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_max_ts']


##MHW Mean Int
MHW_mean_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_mean']
MHW_mean_tr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_mean_tr']
MHW_mean_dtr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_mean_dtr']
MHW_mean_ts_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_mean_ts']


##MHW Cum Int
MHW_cum_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_cum']
MHW_cum_tr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_cum_tr']
MHW_cum_dtr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_cum_dtr']
MHW_cum_ts_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_td']
MHW_td_tr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_td_tr']
MHW_td_dtr_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_td_dtr']
MHW_td_ts_BAL_SAT = TOTAL_MHWs_BAL_SAT['MHW_td_ts']




#########################
## North Atlantic (NA) ##
#########################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_NA_SAT = loadmat(r'...\total_NA_SAT.mat')

#lat and lon
lat_NA_SAT = TOTAL_MHWs_NA_SAT['latitude']
lon_NA_SAT = TOTAL_MHWs_NA_SAT['longitude']
LAT_NA_SAT, LON_NA_SAT = np.meshgrid(lat_NA_SAT, lon_NA_SAT)


##MHW Duration 
MHW_dur_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_dur']
MHW_dur_tr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_dur_tr']
MHW_dur_dtr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_dur_dtr']
MHW_dur_ts_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_dur_ts']


##MHW Frequency
MHW_cnt_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_cnt']
MHW_cnt_tr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_cnt_tr']
MHW_cnt_dtr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_cnt_dtr']
MHW_cnt_ts_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_cnt_ts']


##MHW Max Int
MHW_max_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_max']
MHW_max_tr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_max_tr']
MHW_max_dtr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_max_dtr']
MHW_max_ts_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_max_ts']


##MHW Mean Int
MHW_mean_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_mean']
MHW_mean_tr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_mean_tr']
MHW_mean_dtr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_mean_dtr']
MHW_mean_ts_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_mean_ts']


##MHW Cum Int
MHW_cum_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_cum']
MHW_cum_tr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_cum_tr']
MHW_cum_dtr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_cum_dtr']
MHW_cum_ts_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_td']
MHW_td_tr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_td_tr']
MHW_td_dtr_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_td_dtr']
MHW_td_ts_NA_SAT = TOTAL_MHWs_NA_SAT['MHW_td_ts']




                                ################
                                # GLORYS MODEL #
                                # SURFACE MHWs #
                                ################
                                #   (Annual)

##################
## Canary (CAN) ##
##################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_CAN_MODEL = loadmat(r'...\total_CAN_MODEL.mat')

#lat and lon
lat_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['latitude']
lon_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['longitude']
LAT_CAN_MODEL, LON_CAN_MODEL = np.meshgrid(lat_CAN_MODEL, lon_CAN_MODEL)


##MHW Duration 
MHW_dur_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_dur']
MHW_dur_tr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_dur_tr']
MHW_dur_dtr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_dur_dtr']
MHW_dur_ts_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_dur_ts']


##MHW Frequency
MHW_cnt_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_cnt']
MHW_cnt_tr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_cnt_tr']
MHW_cnt_dtr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_cnt_dtr']
MHW_cnt_ts_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_cnt_ts']

##MHW Max Int
MHW_max_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_max']
MHW_max_tr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_max_tr']
MHW_max_dtr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_max_dtr']
MHW_max_ts_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_max_ts']


##MHW Mean Int
MHW_mean_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_mean']
MHW_mean_tr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_mean_tr']
MHW_mean_dtr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_mean_dtr']
MHW_mean_ts_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_mean_ts']


##MHW Cum Int
MHW_cum_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_cum']
MHW_cum_tr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_cum_tr']
MHW_cum_dtr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_cum_dtr']
MHW_cum_ts_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_td']
MHW_td_tr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_td_tr']
MHW_td_dtr_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_td_dtr']
MHW_td_ts_CAN_MODEL = TOTAL_MHWs_CAN_MODEL['MHW_td_ts']




#########################
## South Atlantic (SA) ##
#########################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_SA_MODEL = loadmat(r'...\total_SA_MODEL.mat')

#lat and lon
lat_SA_MODEL = TOTAL_MHWs_SA_MODEL['latitude']
lon_SA_MODEL = TOTAL_MHWs_SA_MODEL['longitude']
LAT_SA_MODEL, LON_SA_MODEL = np.meshgrid(lat_SA_MODEL, lon_SA_MODEL)


##MHW Duration 
MHW_dur_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_dur']
MHW_dur_tr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_dur_tr']
MHW_dur_dtr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_dur_dtr']
MHW_dur_ts_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_dur_ts']


##MHW Frequency
MHW_cnt_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_cnt']
MHW_cnt_tr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_cnt_tr']
MHW_cnt_dtr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_cnt_dtr']
MHW_cnt_ts_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_cnt_ts']

##MHW Max Int
MHW_max_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_max']
MHW_max_tr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_max_tr']
MHW_max_dtr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_max_dtr']
MHW_max_ts_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_max_ts']


##MHW Mean Int
MHW_mean_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_mean']
MHW_mean_tr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_mean_tr']
MHW_mean_dtr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_mean_dtr']
MHW_mean_ts_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_mean_ts']


##MHW Cum Int
MHW_cum_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_cum']
MHW_cum_tr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_cum_tr']
MHW_cum_dtr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_cum_dtr']
MHW_cum_ts_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_td']
MHW_td_tr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_td_tr']
MHW_td_dtr_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_td_dtr']
MHW_td_ts_SA_MODEL = TOTAL_MHWs_SA_MODEL['MHW_td_ts']




##############################
## SoG and Alboran Sea (AL) ##
##############################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_AL_MODEL = loadmat(r'...\total_AL_MODEL.mat')

#lat and lon
lat_AL_MODEL = TOTAL_MHWs_AL_MODEL['latitude']
lon_AL_MODEL = TOTAL_MHWs_AL_MODEL['longitude']
LAT_AL_MODEL, LON_AL_MODEL = np.meshgrid(lat_AL_MODEL, lon_AL_MODEL)


##MHW Duration 
MHW_dur_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_dur']
MHW_dur_tr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_dur_tr']
MHW_dur_dtr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_dur_dtr']
MHW_dur_ts_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_dur_ts']


##MHW Frequency
MHW_cnt_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_cnt']
MHW_cnt_tr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_cnt_tr']
MHW_cnt_dtr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_cnt_dtr']
MHW_cnt_ts_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_cnt_ts']


##MHW Max Int
MHW_max_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_max']
MHW_max_tr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_max_tr']
MHW_max_dtr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_max_dtr']
MHW_max_ts_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_max_ts']


##MHW Mean Int
MHW_mean_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_mean']
MHW_mean_tr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_mean_tr']
MHW_mean_dtr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_mean_dtr']
MHW_mean_ts_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_mean_ts']


##MHW Cum Int
MHW_cum_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_cum']
MHW_cum_tr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_cum_tr']
MHW_cum_dtr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_cum_dtr']
MHW_cum_ts_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_td']
MHW_td_tr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_td_tr']
MHW_td_dtr_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_td_dtr']
MHW_td_ts_AL_MODEL = TOTAL_MHWs_AL_MODEL['MHW_td_ts']




##############################
## Levantine-Balearic (BAL) ##
##############################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_BAL_MODEL = loadmat(r'...\total_BAL_MODEL.mat')

#lat and lon
lat_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['latitude']
lon_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['longitude']
LAT_BAL_MODEL, LON_BAL_MODEL = np.meshgrid(lat_BAL_MODEL, lon_BAL_MODEL)


##MHW Duration 
MHW_dur_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_dur']
MHW_dur_tr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_dur_tr']
MHW_dur_dtr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_dur_dtr']
MHW_dur_ts_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_dur_ts']


##MHW Frequency
MHW_cnt_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_cnt']
MHW_cnt_tr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_cnt_tr']
MHW_cnt_dtr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_cnt_dtr']
MHW_cnt_ts_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_cnt_ts']


##MHW Max Int
MHW_max_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_max']
MHW_max_tr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_max_tr']
MHW_max_dtr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_max_dtr']
MHW_max_ts_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_max_ts']


##MHW Mean Int
MHW_mean_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_mean']
MHW_mean_tr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_mean_tr']
MHW_mean_dtr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_mean_dtr']
MHW_mean_ts_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_mean_ts']


##MHW Cum Int
MHW_cum_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_cum']
MHW_cum_tr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_cum_tr']
MHW_cum_dtr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_cum_dtr']
MHW_cum_ts_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_td']
MHW_td_tr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_td_tr']
MHW_td_dtr_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_td_dtr']
MHW_td_ts_BAL_MODEL = TOTAL_MHWs_BAL_MODEL['MHW_td_ts']




#########################
## North Atlantic (NA) ##
#########################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_NA_MODEL = loadmat(r'...\total_NA_MODEL.mat')

#lat and lon
lat_NA_MODEL = TOTAL_MHWs_NA_MODEL['latitude']
lon_NA_MODEL = TOTAL_MHWs_NA_MODEL['longitude']
LAT_NA_MODEL, LON_NA_MODEL = np.meshgrid(lat_NA_MODEL, lon_NA_MODEL)


##MHW Duration 
MHW_dur_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_dur']
MHW_dur_tr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_dur_tr']
MHW_dur_dtr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_dur_dtr']
MHW_dur_ts_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_dur_ts']


##MHW Frequency
MHW_cnt_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_cnt']
MHW_cnt_tr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_cnt_tr']
MHW_cnt_dtr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_cnt_dtr']
MHW_cnt_ts_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_cnt_ts']


##MHW Max Int
MHW_max_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_max']
MHW_max_tr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_max_tr']
MHW_max_dtr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_max_dtr']
MHW_max_ts_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_max_ts']


##MHW Mean Int
MHW_mean_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_mean']
MHW_mean_tr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_mean_tr']
MHW_mean_dtr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_mean_dtr']
MHW_mean_ts_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_mean_ts']


##MHW Cum Int
MHW_cum_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_cum']
MHW_cum_tr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_cum_tr']
MHW_cum_dtr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_cum_dtr']
MHW_cum_ts_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_td']
MHW_td_tr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_td_tr']
MHW_td_dtr_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_td_dtr']
MHW_td_ts_NA_MODEL = TOTAL_MHWs_NA_MODEL['MHW_td_ts']




                                ################
                                # GLORYS MODEL #
                                # SURFACE MHWs #
                                ################
                                #  (Monthly)

##################
## Canary (CAN) ##
##################

#Loading the total matrix containing all monthly MHW metrics
TOTAL_MHWs_CAN_MODEL_monthly = loadmat(r'...\total_CAN_MODEL_monthly.mat')


##MHW Duration 
MHW_dur_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_dur']
MHW_dur_ts_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_dur_ts']


##MHW Frequency
MHW_cnt_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_cnt']
MHW_cnt_ts_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_cnt_ts']

##MHW Max Int
MHW_max_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_max']
MHW_max_ts_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_max_ts']


##MHW Mean Int
MHW_mean_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_mean']
MHW_mean_ts_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_mean_ts']


##MHW Cum Int
MHW_cum_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_cum']
MHW_cum_ts_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_cum_ts']


##Total Monthly MHW Days
MHW_td_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_td']
MHW_td_ts_CAN_MODEL_monthly = TOTAL_MHWs_CAN_MODEL_monthly['MHW_td_ts']




#########################
## South Atlantic (SA) ##
#########################

#Loading the total matrix containing all monthly MHW metrics
TOTAL_MHWs_SA_MODEL_monthly = loadmat(r'...\total_SA_MODEL_monthly.mat')


##MHW Duration 
MHW_dur_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_dur']
MHW_dur_ts_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_dur_ts']


##MHW Frequency
MHW_cnt_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_cnt']
MHW_cnt_ts_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_cnt_ts']

##MHW Max Int
MHW_max_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_max']
MHW_max_ts_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_max_ts']


##MHW Mean Int
MHW_mean_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_mean']
MHW_mean_ts_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_mean_ts']


##MHW Cum Int
MHW_cum_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_cum']
MHW_cum_ts_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_cum_ts']


##Total Monthly MHW Days
MHW_td_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_td']
MHW_td_ts_SA_MODEL_monthly = TOTAL_MHWs_SA_MODEL_monthly['MHW_td_ts']




##############################
## SoG and Alboran Sea (AL) ##
##############################

#Loading the total matrix containing all monthly MHW metrics
TOTAL_MHWs_AL_MODEL_monthly = loadmat(r'...\total_AL_MODEL_monthly.mat')


##MHW Duration 
MHW_dur_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_dur']
MHW_dur_ts_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_dur_ts']


##MHW Frequency
MHW_cnt_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_cnt']
MHW_cnt_ts_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_cnt_ts']


##MHW Max Int
MHW_max_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_max']
MHW_max_ts_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_max_ts']


##MHW Mean Int
MHW_mean_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_mean']
MHW_mean_ts_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_mean_ts']


##MHW Cum Int
MHW_cum_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_cum']
MHW_cum_ts_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_cum_ts']


##Total Monthly MHW Days
MHW_td_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_td']
MHW_td_ts_AL_MODEL_monthly = TOTAL_MHWs_AL_MODEL_monthly['MHW_td_ts']




##############################
## Levantine-Balearic (BAL) ##
##############################

#Loading the total matrix containing all monthly MHW metrics
TOTAL_MHWs_BAL_MODEL_monthly = loadmat(r'...\total_BAL_MODEL_monthly.mat')


##MHW Duration 
MHW_dur_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_dur']
MHW_dur_ts_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_dur_ts']


##MHW Frequency
MHW_cnt_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_cnt']
MHW_cnt_ts_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_cnt_ts']


##MHW Max Int
MHW_max_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_max']
MHW_max_ts_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_max_ts']


##MHW Mean Int
MHW_mean_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_mean']
MHW_mean_ts_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_mean_ts']


##MHW Cum Int
MHW_cum_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_cum']
MHW_cum_ts_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_cum_ts']


##Total Monthly MHW Days
MHW_td_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_td']
MHW_td_ts_BAL_MODEL_monthly = TOTAL_MHWs_BAL_MODEL_monthly['MHW_td_ts']




#########################
## North Atlantic (NA) ##
#########################

#Loading the total matrix containing all monthly MHW metrics
TOTAL_MHWs_NA_MODEL_monthly = loadmat(r'...\total_NA_MODEL_monthly.mat')


##MHW Duration 
MHW_dur_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_dur']
MHW_dur_ts_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_dur_ts']


##MHW Frequency
MHW_cnt_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_cnt']
MHW_cnt_ts_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_cnt_ts']


##MHW Max Int
MHW_max_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_max']
MHW_max_ts_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_max_ts']


##MHW Mean Int
MHW_mean_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_mean']
MHW_mean_ts_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_mean_ts']


##MHW Cum Int
MHW_cum_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_cum']
MHW_cum_ts_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_cum_ts']


##Total Monthly MHW Days
MHW_td_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_td']
MHW_td_ts_NA_MODEL_monthly = TOTAL_MHWs_NA_MODEL_monthly['MHW_td_ts']




                                ################
                                # GLORYS MODEL #
                                # BOTTOM MHWs  #
                                ################
                                #   (Annual)

##############
## Canary (CAN) ##
##############

#Loading the total matrix containing all MHW metrics
TOTAL_BMHWs_CAN_MODEL = loadmat(r'...\total_bottom_CAN_MODEL.mat')

#lat and lon
lat_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['latitude']
lon_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['longitude']
LAT_CAN_MODEL, LON_CAN_MODEL = np.meshgrid(lat_CAN_MODEL, lon_CAN_MODEL)


##BMHW Duration 
BMHW_dur_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_dur']
BMHW_dur_tr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_dur_tr']
BMHW_dur_dtr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_dur_dtr']
BMHW_dur_ts_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_cnt']
BMHW_cnt_tr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_cnt_tr']
BMHW_cnt_dtr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_cnt_dtr']
BMHW_cnt_ts_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_cnt_ts']

##BMHW Max Int
BMHW_max_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_max']
BMHW_max_tr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_max_tr']
BMHW_max_dtr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_max_dtr']
BMHW_max_ts_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_mean']
BMHW_mean_tr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_mean_tr']
BMHW_mean_dtr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_mean_dtr']
BMHW_mean_ts_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_cum']
BMHW_cum_tr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_cum_tr']
BMHW_cum_dtr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_cum_dtr']
BMHW_cum_ts_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_cum_ts']


##Total Annual BMHW Days
BMHW_td_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_td']
BMHW_td_tr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_td_tr']
BMHW_td_dtr_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_td_dtr']
BMHW_td_ts_CAN_MODEL = TOTAL_BMHWs_CAN_MODEL['MHW_td_ts']




#########################
## South Atlantic (SA) ##
#########################

#Loading the total matrix containing all BMHW metrics
TOTAL_BMHWs_SA_MODEL = loadmat(r'...\total_bottom_SA_MODEL.mat')

#lat and lon
lat_SA_MODEL = TOTAL_BMHWs_SA_MODEL['latitude']
lon_SA_MODEL = TOTAL_BMHWs_SA_MODEL['longitude']
LAT_SA_MODEL, LON_SA_MODEL = np.meshgrid(lat_SA_MODEL, lon_SA_MODEL)


##BMHW Duration 
BMHW_dur_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_dur']
BMHW_dur_tr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_dur_tr']
BMHW_dur_dtr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_dur_dtr']
BMHW_dur_ts_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_cnt']
BMHW_cnt_tr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_cnt_tr']
BMHW_cnt_dtr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_cnt_dtr']
BMHW_cnt_ts_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_cnt_ts']

##BMHW Max Int
BMHW_max_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_max']
BMHW_max_tr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_max_tr']
BMHW_max_dtr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_max_dtr']
BMHW_max_ts_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_mean']
BMHW_mean_tr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_mean_tr']
BMHW_mean_dtr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_mean_dtr']
BMHW_mean_ts_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_cum']
BMHW_cum_tr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_cum_tr']
BMHW_cum_dtr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_cum_dtr']
BMHW_cum_ts_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_cum_ts']


##Total Annual BMHW Days
BMHW_td_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_td']
BMHW_td_tr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_td_tr']
BMHW_td_dtr_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_td_dtr']
BMHW_td_ts_SA_MODEL = TOTAL_BMHWs_SA_MODEL['MHW_td_ts']




##############################
## SoG and Alboran Sea (AL) ##
##############################

#Loading the total matrix containing all BMHW metrics
TOTAL_BMHWs_AL_MODEL = loadmat(r'...\total_bottom_AL_MODEL.mat')

#lat and lon
lat_AL_MODEL = TOTAL_BMHWs_AL_MODEL['latitude']
lon_AL_MODEL = TOTAL_BMHWs_AL_MODEL['longitude']
LAT_AL_MODEL, LON_AL_MODEL = np.meshgrid(lat_AL_MODEL, lon_AL_MODEL)


##BMHW Duration 
BMHW_dur_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_dur']
BMHW_dur_tr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_dur_tr']
BMHW_dur_dtr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_dur_dtr']
BMHW_dur_ts_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_cnt']
BMHW_cnt_tr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_cnt_tr']
BMHW_cnt_dtr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_cnt_dtr']
BMHW_cnt_ts_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_cnt_ts']


##BMHW Max Int
BMHW_max_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_max']
BMHW_max_tr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_max_tr']
BMHW_max_dtr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_max_dtr']
BMHW_max_ts_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_mean']
BMHW_mean_tr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_mean_tr']
BMHW_mean_dtr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_mean_dtr']
BMHW_mean_ts_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_cum']
BMHW_cum_tr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_cum_tr']
BMHW_cum_dtr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_cum_dtr']
BMHW_cum_ts_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_cum_ts']


##Total Annual BMHW Days
BMHW_td_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_td']
BMHW_td_tr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_td_tr']
BMHW_td_dtr_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_td_dtr']
BMHW_td_ts_AL_MODEL = TOTAL_BMHWs_AL_MODEL['MHW_td_ts']




##############################
## Levantine-Balearic (BAL) ##
##############################

#Loading the total matrix containing all BMHW metrics
TOTAL_BMHWs_BAL_MODEL = loadmat(r'...\total_bottom_BAL_MODEL.mat')

#lat and lon
lat_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['latitude']
lon_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['longitude']
LAT_BAL_MODEL, LON_BAL_MODEL = np.meshgrid(lat_BAL_MODEL, lon_BAL_MODEL)


##BMHW Duration 
BMHW_dur_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_dur']
BMHW_dur_tr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_dur_tr']
BMHW_dur_dtr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_dur_dtr']
BMHW_dur_ts_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_cnt']
BMHW_cnt_tr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_cnt_tr']
BMHW_cnt_dtr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_cnt_dtr']
BMHW_cnt_ts_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_cnt_ts']


##BMHW Max Int
BMHW_max_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_max']
BMHW_max_tr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_max_tr']
BMHW_max_dtr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_max_dtr']
BMHW_max_ts_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_mean']
BMHW_mean_tr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_mean_tr']
BMHW_mean_dtr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_mean_dtr']
BMHW_mean_ts_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_cum']
BMHW_cum_tr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_cum_tr']
BMHW_cum_dtr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_cum_dtr']
BMHW_cum_ts_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_cum_ts']


##Total Annual BMHW Days
BMHW_td_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_td']
BMHW_td_tr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_td_tr']
BMHW_td_dtr_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_td_dtr']
BMHW_td_ts_BAL_MODEL = TOTAL_BMHWs_BAL_MODEL['MHW_td_ts']




#########################
## North Atlantic (NA) ##
#########################

#Loading the total matrix containing all BMHW metrics
TOTAL_BMHWs_NA_MODEL = loadmat(r'...\total_bottom_NA_MODEL.mat')

#lat and lon
lat_NA_MODEL = TOTAL_BMHWs_NA_MODEL['latitude']
lon_NA_MODEL = TOTAL_BMHWs_NA_MODEL['longitude']
LAT_NA_MODEL, LON_NA_MODEL = np.meshgrid(lat_NA_MODEL, lon_NA_MODEL)


##BMHW Duration 
BMHW_dur_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_dur']
BMHW_dur_tr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_dur_tr']
BMHW_dur_dtr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_dur_dtr']
BMHW_dur_ts_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_cnt']
BMHW_cnt_tr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_cnt_tr']
BMHW_cnt_dtr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_cnt_dtr']
BMHW_cnt_ts_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_cnt_ts']


##BMHW Max Int
BMHW_max_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_max']
BMHW_max_tr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_max_tr']
BMHW_max_dtr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_max_dtr']
BMHW_max_ts_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_mean']
BMHW_mean_tr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_mean_tr']
BMHW_mean_dtr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_mean_dtr']
BMHW_mean_ts_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_cum']
BMHW_cum_tr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_cum_tr']
BMHW_cum_dtr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_cum_dtr']
BMHW_cum_ts_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_cum_ts']


##Total Annual BMHW Days
BMHW_td_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_td']
BMHW_td_tr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_td_tr']
BMHW_td_dtr_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_td_dtr']
BMHW_td_ts_NA_MODEL = TOTAL_BMHWs_NA_MODEL['MHW_td_ts']




                                ################
                                # GLORYS MODEL #
                                # BOTTOM MHWs  #
                                ################
                                #   (Monthly)

##############
## Canary (CAN) ##
##############

#Loading the total matrix containing all mothly BMHW metrics
TOTAL_BMHWs_CAN_MODEL_monthly = loadmat(r'...\total_bottom_CAN_MODEL_monthly.mat')


##BMHW Duration 
BMHW_dur_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_dur']
BMHW_dur_ts_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_cnt']
BMHW_cnt_ts_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_cnt_ts']

##BMHW Max Int
BMHW_max_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_max']
BMHW_max_ts_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_mean']
BMHW_mean_ts_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_cum']
BMHW_cum_ts_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_cum_ts']


##Total Monthly BMHW Days
BMHW_td_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_td']
BMHW_td_ts_CAN_MODEL_monthly = TOTAL_BMHWs_CAN_MODEL_monthly['MHW_td_ts']




#########################
## South Atlantic (SA) ##
#########################

#Loading the total matrix containing all mothly BMHW metrics
TOTAL_BMHWs_SA_MODEL_monthly = loadmat(r'...\total_bottom_SA_MODEL_monthly.mat')


##BMHW Duration 
BMHW_dur_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_dur']
BMHW_dur_ts_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_cnt']
BMHW_cnt_ts_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_cnt_ts']


##BMHW Max Int
BMHW_max_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_max']
BMHW_max_ts_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_mean']
BMHW_mean_ts_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_cum']
BMHW_cum_ts_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_cum_ts']


##Total Monthly BMHW Days
BMHW_td_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_td']
BMHW_td_ts_SA_MODEL_monthly = TOTAL_BMHWs_SA_MODEL_monthly['MHW_td_ts']




##############################
## SoG and Alboran Sea (AL) ##
##############################

#Loading the total matrix containing all mothly BMHW metrics
TOTAL_BMHWs_AL_MODEL_monthly = loadmat(r'...\total_bottom_AL_MODEL_monthly.mat')


##BMHW Duration 
BMHW_dur_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_dur']
BMHW_dur_ts_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_cnt']
BMHW_cnt_ts_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_cnt_ts']


##BMHW Max Int
BMHW_max_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_max']
BMHW_max_ts_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_mean']
BMHW_mean_ts_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_cum']
BMHW_cum_ts_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_cum_ts']


##Total Monthly BMHW Days
BMHW_td_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_td']
BMHW_td_ts_AL_MODEL_monthly = TOTAL_BMHWs_AL_MODEL_monthly['MHW_td_ts']




##############################
## Levantine-Balearic (BAL) ##
##############################

#Loading the total matrix containing all mothly BMHW metrics
TOTAL_BMHWs_BAL_MODEL_monthly = loadmat(r'...\total_bottom_BAL_MODEL_monthly.mat')


##BMHW Duration 
BMHW_dur_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_dur']
BMHW_dur_ts_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_cnt']
BMHW_cnt_ts_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_cnt_ts']


##BMHW Max Int
BMHW_max_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_max']
BMHW_max_ts_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_mean']
BMHW_mean_ts_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_cum']
BMHW_cum_ts_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_cum_ts']


##Total Monthly BMHW Days
BMHW_td_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_td']
BMHW_td_ts_BAL_MODEL_monthly = TOTAL_BMHWs_BAL_MODEL_monthly['MHW_td_ts']




#########################
## North Atlantic (NA) ##
#########################

#Loading the total matrix containing all mothly BMHW metrics
TOTAL_BMHWs_NA_MODEL_monthly = loadmat(r'...\total_bottom_NA_MODEL_monthly.mat')


##BMHW Duration 
BMHW_dur_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_dur']
BMHW_dur_ts_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_cnt']
BMHW_cnt_ts_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_cnt_ts']


##BMHW Max Int
BMHW_max_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_max']
BMHW_max_ts_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_mean']
BMHW_mean_ts_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_cum']
BMHW_cum_ts_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_cum_ts']


##Total Monthly BMHW Days
BMHW_td_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_td']
BMHW_td_ts_NA_MODEL_monthly = TOTAL_BMHWs_NA_MODEL_monthly['MHW_td_ts']




                #####################
                ## Bathymetry Data ##
                #####################

ds_BAT_CAN = xr.open_dataset(r'.../Bathy_Canary (CAN)_clipped.nc')
elevation_CAN = ds_BAT_CAN['elevation'] * (-1)
elevation_CAN = xr.where(elevation_CAN <= 0, np.NaN, elevation_CAN)
lon_CAN_bat = ds_BAT_CAN['lon']
lat_CAN_bat = ds_BAT_CAN['lat']
LON_CAN_bat, LAT_CAN_bat = np.meshgrid(lon_CAN_bat, lat_CAN_bat)

ds_BAT_SA = xr.open_dataset(r'.../Bathy_SA_clipped.nc')
elevation_SA = ds_BAT_SA['elevation'] * (-1)
elevation_SA = xr.where(elevation_SA <= 0, np.NaN, elevation_SA)
lon_SA_bat = ds_BAT_SA['lon']
lat_SA_bat = ds_BAT_SA['lat']
LON_SA_bat, LAT_SA_bat = np.meshgrid(lon_SA_bat, lat_SA_bat)

ds_BAT_AL = xr.open_dataset(r'.../Bathy_AL_clipped.nc')
elevation_AL = ds_BAT_AL['elevation'] * (-1)
elevation_AL = xr.where(elevation_AL <= 0, np.NaN, elevation_AL)
lon_AL_bat = ds_BAT_AL['lon']
lat_AL_bat = ds_BAT_AL['lat']
LON_AL_bat, LAT_AL_bat = np.meshgrid(lon_AL_bat, lat_AL_bat)

ds_BAT_BAL = xr.open_dataset(r'.../Bathy_BAL_clipped.nc')
elevation_BAL = ds_BAT_BAL['elevation'] * (-1)
elevation_BAL = xr.where(elevation_BAL <= 0, np.NaN, elevation_BAL)
lon_BAL_bat = ds_BAT_BAL['lon']
lat_BAL_bat = ds_BAT_BAL['lat']
LON_BAL_bat, LAT_BAL_bat = np.meshgrid(lon_BAL_bat, lat_BAL_bat)

ds_BAT_NA = xr.open_dataset(r'.../Bathy_NA_clipped.nc')
elevation_NA = ds_BAT_NA['elevation'] * (-1)
elevation_NA = xr.where(elevation_NA <= 0, np.NaN, elevation_NA)
lon_NA_bat = ds_BAT_NA['lon']
lat_NA_bat = ds_BAT_NA['lat']
LON_NA_bat, LAT_NA_bat = np.meshgrid(lon_NA_bat, lat_NA_bat)




                ##########################
                ## GLORYS 12V1 Datasets ##
                ##########################

#Load the previously-clipped dataset
ds_Model_CAN = xr.open_dataset(r'...\Temp_MLD_GLORYS_Canary_clipped.nc')
ds_Model_SA = xr.open_dataset(r'...\Temp_MLD_GLORYS_SA_clipped.nc')
ds_Model_AL = xr.open_dataset(r'...\Temp_MLD_GLORYS_AL_clipped.nc')
ds_Model_BAL = xr.open_dataset(r'...\MLD_GLORYS_BAL_clipped.nc')
ds_Model_NA = xr.open_dataset(r'...\MLD_GLORYS_NA_clipped.nc')
