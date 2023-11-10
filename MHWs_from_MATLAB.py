# -*- coding: utf-8 -*-
"""
##################################.mat to .npy##################################
"""

from scipy.io import loadmat
import numpy as np
import xarray as xr 

                                #############
                                # SATELLITE #
                                # 1982-2012 #
                                #############
                                
##############
## Canarias ##
##############

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_CAN_SAT_1 = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Canarias\total_canarias_1.mat')

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




#################
## Golfo Cadiz ##
#################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_GC_SAT_1 = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Golfo_Cadiz\total_GC_1.mat')

#lat and lon
lat_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['latitude']
lon_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['longitude']
LAT_GC_SAT_1, LON_GC_SAT_1 = np.meshgrid(lat_GC_SAT_1, lon_GC_SAT_1)


##MHW Duration 
MHW_dur_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_dur']
MHW_dur_tr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_dur_tr']
MHW_dur_dtr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_dur_dtr']
MHW_dur_ts_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_dur_ts']


##MHW Frequency
MHW_cnt_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_cnt']
MHW_cnt_tr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_cnt_tr']
MHW_cnt_dtr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_cnt_dtr']
MHW_cnt_ts_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_cnt_ts']

##MHW Max Int
MHW_max_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_max']
MHW_max_tr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_max_tr']
MHW_max_dtr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_max_dtr']
MHW_max_ts_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_max_ts']


##MHW Mean Int
MHW_mean_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_mean']
MHW_mean_tr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_mean_tr']
MHW_mean_dtr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_mean_dtr']
MHW_mean_ts_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_mean_ts']


##MHW Cum Int
MHW_cum_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_cum']
MHW_cum_tr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_cum_tr']
MHW_cum_dtr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_cum_dtr']
MHW_cum_ts_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_td']
MHW_td_tr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_td_tr']
MHW_td_dtr_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_td_dtr']
MHW_td_ts_GC_SAT_1 = TOTAL_MHWs_GC_SAT_1['MHW_td_ts']





#############
## Alboran ##
#############

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_AL_SAT_1 = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Alboran\total_AL_1.mat')

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





######################
## Levantino-balear ##
######################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_BAL_SAT_1 = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Baleares\total_BAL_1.mat')

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




##################
## Noratlántica ##
##################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_NA_SAT_1 = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Noratlantica\total_NA_1.mat')

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
                                
##############
## Canarias ##
##############

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_CAN_SAT = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Canarias\total_CAN_SAT.mat')

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




#################
## Golfo Cadiz ##
#################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_GC_SAT = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Golfo_Cadiz\total_GC_SAT.mat')

#lat and lon
lat_GC_SAT = TOTAL_MHWs_GC_SAT['latitude']
lon_GC_SAT = TOTAL_MHWs_GC_SAT['longitude']
LAT_GC_SAT, LON_GC_SAT = np.meshgrid(lat_GC_SAT, lon_GC_SAT)


##MHW Duration 
MHW_dur_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_dur']
MHW_dur_tr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_dur_tr']
MHW_dur_dtr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_dur_dtr']
MHW_dur_ts_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_dur_ts']


##MHW Frequency
MHW_cnt_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_cnt']
MHW_cnt_tr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_cnt_tr']
MHW_cnt_dtr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_cnt_dtr']
MHW_cnt_ts_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_cnt_ts']

##MHW Max Int
MHW_max_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_max']
MHW_max_tr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_max_tr']
MHW_max_dtr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_max_dtr']
MHW_max_ts_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_max_ts']


##MHW Mean Int
MHW_mean_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_mean']
MHW_mean_tr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_mean_tr']
MHW_mean_dtr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_mean_dtr']
MHW_mean_ts_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_mean_ts']


##MHW Cum Int
MHW_cum_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_cum']
MHW_cum_tr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_cum_tr']
MHW_cum_dtr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_cum_dtr']
MHW_cum_ts_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_td']
MHW_td_tr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_td_tr']
MHW_td_dtr_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_td_dtr']
MHW_td_ts_GC_SAT = TOTAL_MHWs_GC_SAT['MHW_td_ts']





#############
## Alboran ##
#############

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_AL_SAT = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Alboran\total_AL_SAT.mat')

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





######################
## Levantino-balear ##
######################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_BAL_SAT = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Baleares\total_BAL_SAT.mat')

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




##################
## Noratlántica ##
##################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_NA_SAT = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Noratlantica\total_NA_SAT.mat')

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
                                ################
                                
##############
## Canarias ##
##############

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_CAN_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Canarias\total_CAN_MODEL.mat')

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




#################
## Golfo Cadiz ##
#################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_GC_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Golfo_Cadiz\total_GC_MODEL.mat')

#lat and lon
lat_GC_MODEL = TOTAL_MHWs_GC_MODEL['latitude']
lon_GC_MODEL = TOTAL_MHWs_GC_MODEL['longitude']
LAT_GC_MODEL, LON_GC_MODEL = np.meshgrid(lat_GC_MODEL, lon_GC_MODEL)


##MHW Duration 
MHW_dur_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_dur']
MHW_dur_tr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_dur_tr']
MHW_dur_dtr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_dur_dtr']
MHW_dur_ts_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_dur_ts']


##MHW Frequency
MHW_cnt_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_cnt']
MHW_cnt_tr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_cnt_tr']
MHW_cnt_dtr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_cnt_dtr']
MHW_cnt_ts_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_cnt_ts']

##MHW Max Int
MHW_max_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_max']
MHW_max_tr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_max_tr']
MHW_max_dtr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_max_dtr']
MHW_max_ts_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_max_ts']


##MHW Mean Int
MHW_mean_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_mean']
MHW_mean_tr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_mean_tr']
MHW_mean_dtr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_mean_dtr']
MHW_mean_ts_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_mean_ts']


##MHW Cum Int
MHW_cum_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_cum']
MHW_cum_tr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_cum_tr']
MHW_cum_dtr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_cum_dtr']
MHW_cum_ts_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_cum_ts']


##Total Annual MHW Days
MHW_td_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_td']
MHW_td_tr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_td_tr']
MHW_td_dtr_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_td_dtr']
MHW_td_ts_GC_MODEL = TOTAL_MHWs_GC_MODEL['MHW_td_ts']





#############
## Alboran ##
#############

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_AL_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Alboran\total_AL_MODEL.mat')

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





######################
## Levantino-balear ##
######################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_BAL_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Baleares\total_BAL_MODEL.mat')

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




##################
## Noratlántica ##
##################

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_NA_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Noratlantica\total_NA_MODEL.mat')

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
                                # BOTTOM MHWs  #
                                ################
                                
##############
## Canarias ##
##############

#Loading the total matrix containing all MHW metrics
TOTAL_BMHWs_CAN_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Canarias\total_bottom_CAN_MODEL.mat')

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




#################
## Golfo Cadiz ##
#################

#Loading the total matrix containing all BMHW metrics
TOTAL_BMHWs_GC_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Golfo_Cadiz\total_bottom_GC_MODEL.mat')

#lat and lon
lat_GC_MODEL = TOTAL_BMHWs_GC_MODEL['latitude']
lon_GC_MODEL = TOTAL_BMHWs_GC_MODEL['longitude']
LAT_GC_MODEL, LON_GC_MODEL = np.meshgrid(lat_GC_MODEL, lon_GC_MODEL)


##BMHW Duration 
BMHW_dur_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_dur']
BMHW_dur_tr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_dur_tr']
BMHW_dur_dtr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_dur_dtr']
BMHW_dur_ts_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_dur_ts']


##BMHW Frequency
BMHW_cnt_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_cnt']
BMHW_cnt_tr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_cnt_tr']
BMHW_cnt_dtr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_cnt_dtr']
BMHW_cnt_ts_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_cnt_ts']

##BMHW Max Int
BMHW_max_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_max']
BMHW_max_tr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_max_tr']
BMHW_max_dtr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_max_dtr']
BMHW_max_ts_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_max_ts']


##BMHW Mean Int
BMHW_mean_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_mean']
BMHW_mean_tr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_mean_tr']
BMHW_mean_dtr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_mean_dtr']
BMHW_mean_ts_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_mean_ts']


##BMHW Cum Int
BMHW_cum_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_cum']
BMHW_cum_tr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_cum_tr']
BMHW_cum_dtr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_cum_dtr']
BMHW_cum_ts_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_cum_ts']


##Total Annual BMHW Days
BMHW_td_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_td']
BMHW_td_tr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_td_tr']
BMHW_td_dtr_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_td_dtr']
BMHW_td_ts_GC_MODEL = TOTAL_BMHWs_GC_MODEL['MHW_td_ts']





#############
## Alboran ##
#############

#Loading the total matrix containing all BMHW metrics
TOTAL_BMHWs_AL_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Alboran\total_bottom_AL_MODEL.mat')

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





######################
## Levantino-balear ##
######################

#Loading the total matrix containing all BMHW metrics
TOTAL_BMHWs_BAL_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Baleares\total_bottom_BAL_MODEL.mat')

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




##################
## Noratlántica ##
##################

#Loading the total matrix containing all BMHW metrics
TOTAL_BMHWs_NA_MODEL = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Noratlantica\total_bottom_NA_MODEL.mat')

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





                #####################
                ## Bathymetry Data ##
                #####################


ds_BAT_canarias = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_Bathymetry\Canarias/Bathy_Canarias_clipped.nc')
elevation_CAN = ds_BAT_canarias['elevation'] * (-1)
elevation_CAN = xr.where(elevation_CAN <= 0, np.NaN, elevation_CAN)
lon_CAN_bat = ds_BAT_canarias['lon']
lat_CAN_bat = ds_BAT_canarias['lat']
LON_CAN_bat, LAT_CAN_bat = np.meshgrid(lon_CAN_bat, lat_CAN_bat)

ds_BAT_GC = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_Bathymetry\GC/Bathy_GC_clipped.nc')
elevation_GC = ds_BAT_GC['elevation'] * (-1)
elevation_GC = xr.where(elevation_GC <= 0, np.NaN, elevation_GC)
lon_GC_bat = ds_BAT_GC['lon']
lat_GC_bat = ds_BAT_GC['lat']
LON_GC_bat, LAT_GC_bat = np.meshgrid(lon_GC_bat, lat_GC_bat)

ds_BAT_AL = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_Bathymetry\AL/Bathy_AL_clipped.nc')
elevation_AL = ds_BAT_AL['elevation'] * (-1)
elevation_AL = xr.where(elevation_AL <= 0, np.NaN, elevation_AL)
lon_AL_bat = ds_BAT_AL['lon']
lat_AL_bat = ds_BAT_AL['lat']
LON_AL_bat, LAT_AL_bat = np.meshgrid(lon_AL_bat, lat_AL_bat)

ds_BAT_BAL = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_Bathymetry\BAL/Bathy_BAL_clipped.nc')
elevation_BAL = ds_BAT_BAL['elevation'] * (-1)
elevation_BAL = xr.where(elevation_BAL <= 0, np.NaN, elevation_BAL)
lon_BAL_bat = ds_BAT_BAL['lon']
lat_BAL_bat = ds_BAT_BAL['lat']
LON_BAL_bat, LAT_BAL_bat = np.meshgrid(lon_BAL_bat, lat_BAL_bat)

ds_BAT_NA = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_Bathymetry\NA/Bathy_NA_clipped.nc')
elevation_NA = ds_BAT_NA['elevation'] * (-1)
elevation_NA = xr.where(elevation_NA <= 0, np.NaN, elevation_NA)
lon_NA_bat = ds_BAT_NA['lon']
lat_NA_bat = ds_BAT_NA['lat']
LON_NA_bat, LAT_NA_bat = np.meshgrid(lon_NA_bat, lat_NA_bat)



                ##########################
                ## GLORYS 12V1 Datasets ##
                ##########################

#Load the previously-clipped dataset
ds_Model_CAN = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\Canarias\Temp_MLD_GLORYS_Canary_clipped.nc')
ds_Model_GC = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\GC\Temp_MLD_GLORYS_GC_clipped.nc')
ds_Model_AL = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\AL\Temp_MLD_GLORYS_AL_clipped.nc')
ds_Model_BAL = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\BAL\MLD_GLORYS_BAL_clipped.nc')
ds_Model_NA = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\NA\MLD_GLORYS_NA_clipped.nc')

















