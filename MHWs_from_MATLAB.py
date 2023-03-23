# -*- coding: utf-8 -*-
"""
##################################.mat to .npy##################################
"""

from scipy.io import loadmat
import numpy as np
from netCDF4 import Dataset 



##############
## Canarias ##
##############

#Loading the total matrix containing all MHW metrics
TOTAL_MHWs_Canary = loadmat(r'E:\ICMAN-CSIC\Estrategias_Marinas\MHWs\Canarias\total.mat')

#lat and lon
lat_canarias = TOTAL_MHWs_Canary['latitude']
lon_canarias = TOTAL_MHWs_Canary['longitude']
LAT_canarias, LON_canarias = np.meshgrid(lat_canarias, lon_canarias)


##MHW Duration 
MHW_dur_canarias = TOTAL_MHWs_Canary['MHW_dur']
MHW_dur_tr_canarias = TOTAL_MHWs_Canary['MHW_dur_tr']
MHW_dur_dtr_canarias = TOTAL_MHWs_Canary['MHW_dur_dtr']
MHW_dur_ts_canarias = TOTAL_MHWs_Canary['MHW_dur_ts']


##MHW Frequency
MHW_cnt_canarias = TOTAL_MHWs_Canary['MHW_cnt']
MHW_cnt_tr_canarias = TOTAL_MHWs_Canary['MHW_cnt_tr']
MHW_cnt_dtr_canarias = TOTAL_MHWs_Canary['MHW_cnt_dtr']
MHW_cnt_ts_canarias = TOTAL_MHWs_Canary['MHW_cnt_ts']

##MHW Max Int
MHW_max_canarias = TOTAL_MHWs_Canary['MHW_max']
MHW_max_tr_canarias = TOTAL_MHWs_Canary['MHW_max_tr']
MHW_max_dtr_canarias = TOTAL_MHWs_Canary['MHW_max_dtr']
MHW_max_ts_canarias = TOTAL_MHWs_Canary['MHW_max_ts']


##MHW Mean Int
MHW_mean_canarias = TOTAL_MHWs_Canary['MHW_mean']
MHW_mean_tr_canarias = TOTAL_MHWs_Canary['MHW_mean_tr']
MHW_mean_dtr_canarias = TOTAL_MHWs_Canary['MHW_mean_dtr']
MHW_mean_ts_canarias = TOTAL_MHWs_Canary['MHW_mean_ts']


##MHW Cum Int
MHW_cum_canarias = TOTAL_MHWs_Canary['MHW_cum']
MHW_cum_tr_canarias = TOTAL_MHWs_Canary['MHW_cum_tr']
MHW_cum_dtr_canarias = TOTAL_MHWs_Canary['MHW_cum_dtr']
MHW_cum_ts_canarias = TOTAL_MHWs_Canary['MHW_cum_ts']


##MHW Cum Int
MHW_td_canarias = TOTAL_MHWs_Canary['MHW_td']
MHW_td_tr_canarias = TOTAL_MHWs_Canary['MHW_td_tr']
MHW_td_dtr_canarias = TOTAL_MHWs_Canary['MHW_td_dtr']
MHW_td_ts_canarias = TOTAL_MHWs_Canary['MHW_td_ts']











