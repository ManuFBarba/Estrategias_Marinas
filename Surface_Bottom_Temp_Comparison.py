# -*- coding: utf-8 -*-
"""

##################### Surface & Bottom Temperatures Comparison ################

"""

#Loading required python modules
from scipy.io import loadmat
import pandas as pd
import numpy as np
import xarray as xr 

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
from scipy import stats

import datetime
from datetime import date
import ecoliver as ecj


##Load MHWs_from_MATLAB.py##

##Load the previously-clipped SST dataset
#North Atlantic
ds_NA = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Noratlantica/SST_NA_clipped.nc')
sst_NA = ds_NA['analysed_sst'] - 273.15 #K to ºC
ds_Model_NA = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\NA\T_GLORYS_NA_clipped.nc')
thetao_NA = ds_Model_NA['thetao']
ds_Model_NA = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\NA\bottomT_GLORYS_NA_clipped.nc')
bottomT_NA = ds_Model_NA['bottomT']

#SoG and Alboran
ds_AL = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Alboran/SST_AL_clipped.nc')
sst_AL = ds_AL['analysed_sst'] - 273.15 #K to ºC
ds_Model_AL = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\AL\Temp_MLD_GLORYS_AL_clipped.nc')
thetao_AL = ds_Model_AL['thetao']
bottomT_AL = ds_Model_AL['bottomT']

#Canary
ds_canarias = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Canarias/SST_Canary_clipped.nc')
sst_canarias = ds_canarias['analysed_sst'] - 273.15 #K to ºC
ds_Model_CAN = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\Canarias\Temp_MLD_GLORYS_Canary_clipped.nc')
thetao_CAN = ds_Model_CAN['thetao']
bottomT_CAN = ds_Model_CAN['bottomT']

#South Atlantic
ds_GC = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Golfo_Cadiz/SST_GC_clipped.nc')
sst_GC = ds_GC['analysed_sst'] - 273.15 #K to ºC
ds_Model_GC = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\GC\Temp_MLD_GLORYS_GC_clipped.nc')
thetao_GC = ds_Model_GC['thetao']
bottomT_GC = ds_Model_GC['bottomT']

#Levantine-Balearic
ds_BAL = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas\Datos_SST\Baleares/SST_BAL_clipped.nc')
sst_BAL = ds_BAL['analysed_sst'] - 273.15 #K to ºC
ds_Model_BAL = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\BAL\T_GLORYS_BAL_clipped.nc')
thetao_BAL = ds_Model_BAL['thetao']
ds_Model_BAL = xr.open_dataset(r'E:\ICMAN-CSIC\Estrategias_Marinas/Numerical_Model_GLORYS\BAL\bottomT_GLORYS_BAL_clipped.nc')
bottomT_BAL = ds_Model_BAL['bottomT']



                    ######################################
                    ### SURFACE TEMPERATURE COMPARISON ###
                    ######################################

                            ####################
                            ## North Atlantic ##
                            ####################

## Timeseries of surface temperature
surface_NA_SAT_ts = np.nanmean(sst_NA, axis=(1,2))
surface_NA_MODEL_ts = np.nanmean(thetao_NA, axis=(1,2))
bottom_NA_MODEL_ts = np.nanmean(bottomT_NA, axis=(1,2))

#Mask Satellite data prior to 1993 and/or instrument errors
surface_NA_SAT_ts[0:4018] = np.NaN
surface_NA_SAT_ts[13514:13878] = np.NaN

##Remove NaN values
# surface_NA_SAT_ts = surface_NA_SAT_ts[~np.isnan(surface_NA_SAT_ts)]
##Remove outliers: outlier > 3* std
# mean = np.mean(surface_NA_SAT_ts)
# std = np.std(surface_NA_SAT_ts)
# threshold = 3 * std
# surface_NA_SAT_ts = surface_NA_SAT_ts[abs(surface_NA_SAT_ts - mean) < threshold]


##Standard error surface temperature
# error_surface_NA_SAT = (np.nanstd(sst_NA, axis=(1,2)))/np.sqrt(14975)
# error_surface_NA_MODEL = (np.nanstd(thetao_NA, axis=(1,2)))/np.sqrt(10957)


#Times from 01-01-1982 to 31-12-2022 and from 01-01-1993 to 31-12-2022
t, dates_1, T, year, month, day, doy = ecj.timevector([1982, 1, 1], [2022, 12, 31])
t, dates_2, T, year, month, day, doy = ecj.timevector([1993, 1, 1], [2022, 12, 31])

ts = date(1992,7,31)
te = date(2023,7,31)



fig, axs = plt.subplots(figsize=(20, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})

# time_1 = np.arange(1, len(surface_NA_SAT_ts)+1)
# time_2 = np.arange(1, len(surface_NA_MODEL_ts)+1)

# #Surface temperature rating change per decade
# res_surface_NA_SAT = stats.linregress(time_1, surface_NA_SAT_ts)
# res_surface_NA_MODEL = stats.linregress(time_2, surface_NA_MODEL_ts)
# res_bottom_NA_MODEL = stats.linregress(time_2, bottom_NA_MODEL_ts)

# Calculate the root mean square error (RMSE) between the satellite and model time series
# Find the index corresponding to the start of the overlapping period
# start_index = np.where(np.array(dates_1) >= datetime.date(1993, 1, 1))[0][0]
# # Trim the satellite time series to match the length of the model time series
# trimmed_surface_NA_SAT_ts = surface_NA_SAT_ts[start_index:]
# rmse = np.sqrt(np.mean((trimmed_surface_NA_SAT_ts - surface_NA_MODEL_ts) ** 2))

# print("Estimated temperature change per decade (Satellite):", (res_surface_NA_SAT.slope)*365*10)
# print("Estimated temperature change per decade (Model):", (res_surface_NA_MODEL.slope)*365*10)
# print("Estimated bottom temperature change per decade (Model):", (res_bottom_NA_MODEL.slope)*365*10)
# print("Estimated RMSE SAT/MODEL:", rmse)

axs.plot(dates_1, surface_NA_SAT_ts, '-', color='black', linewidth=1.5, label='Surface IBI - ODYSSEA L4')

axs.plot(dates_2, surface_NA_MODEL_ts, '-', color='red', linewidth=1.5, label='Surface GLORYS12V1')

bottom_ax = axs.twinx()
bottom_ax.plot(dates_2, bottom_NA_MODEL_ts, '-', color='xkcd:mango', alpha=0.7, linewidth=1.5, label='Bottom GLORYS12V1')


axs.tick_params(length=10, direction='in')
axs.set_ylim(10, 24)
axs.set_yticks([12, 14, 16, 18, 20, 22])
axs.set_ylabel('Surface temperature [$^\circ$C]')
axs.set_xlim(ts, te)
# axs.set_xticks(['1995', '2000', '2005', '2010', '2015', '2020'])
axs.set_title('a     North Atlantic (NA)')

bottom_ax.tick_params(length=10, direction='in')
bottom_ax.set_ylim(3.5, 4.25)  
bottom_ax.set_yticks([3.5, 3.6, 3.7, 3.8])  
bottom_ax.set_ylabel('Bottom temperature [$^\circ$C]')  
bottom_ax.spines['right'].set_color('xkcd:mango')
bottom_ax.yaxis.label.set_color('xkcd:mango')
bottom_ax.tick_params(axis='y', colors='xkcd:mango')  
bottom_ax.yaxis.set_tick_params(labelcolor='xkcd:mango')


# fig.legend(loc=(0.13, 0.73), frameon=False, fontsize=14)
fig.legend(loc=(0.08, 0.77), ncol=3, frameon=False, fontsize=18)
# fig.legend(loc='upper left', frameon=False, fontsize=14)



outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\NA_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





                            #########################
                            ## SoG and Alboran Sea ##
                            #########################


## Timeseries of surface temperature
surface_AL_SAT_ts = np.nanmean(sst_AL, axis=(1,2))
surface_AL_MODEL_ts = np.nanmean(thetao_AL, axis=(1,2))
bottom_AL_MODEL_ts = np.nanmean(bottomT_AL, axis=(1,2))

#Mask Satellite data prior to 1993 and/or instrument errors
surface_AL_SAT_ts[0:4018] = np.NaN
surface_AL_SAT_ts[13514:13878] = np.NaN

##Remove NaN values
# surface_AL_SAT_ts = surface_AL_SAT_ts[~np.isnan(surface_AL_SAT_ts)]
##Remove outliers: outlier > 3* std
# mean = np.mean(surface_AL_SAT_ts)
# std = np.std(surface_AL_SAT_ts)
# threshold = 3 * std
# surface_AL_SAT_ts = surface_AL_SAT_ts[abs(surface_AL_SAT_ts - mean) < threshold]


##Standard error surface temperature
# error_surface_AL_SAT = (np.nanstd(sst_AL, axis=(1,2)))/np.sqrt(14975)
# error_surface_AL_MODEL = (np.nanstd(thetao_AL, axis=(1,2)))/np.sqrt(10957)


#Times from 01-01-1982 to 31-12-2022 and from 01-01-1993 to 31-12-2022
t, dates_1, T, year, month, day, doy = ecj.timevector([1982, 1, 1], [2022, 12, 31])
t, dates_2, T, year, month, day, doy = ecj.timevector([1993, 1, 1], [2022, 12, 31])

ts = date(1992,7,31)
te = date(2023,7,31)



fig, axs = plt.subplots(figsize=(20, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})

# time_1 = np.arange(1, len(surface_AL_SAT_ts)+1)
# time_2 = np.arange(1, len(surface_AL_MODEL_ts)+1)

# #Surface temperature rating change per decade
# res_surface_AL_SAT = stats.linregress(time_1, surface_AL_SAT_ts)
# res_surface_AL_MODEL = stats.linregress(time_2, surface_AL_MODEL_ts)
# res_bottom_AL_MODEL = stats.linregress(time_2, bottom_AL_MODEL_ts)

# Calculate the root mean square error (RMSE) between the satellite and model time series
# Find the index corresponding to the start of the overlapping period
# start_index = np.where(np.array(dates_1) >= datetime.date(1993, 1, 1))[0][0]
# # Trim the satellite time series to match the length of the model time series
# trimmed_surface_AL_SAT_ts = surface_AL_SAT_ts[start_index:]
# rmse = np.sqrt(np.mean((trimmed_surface_AL_SAT_ts - surface_AL_MODEL_ts) ** 2))

# print("Estimated temperature change per decade (Satellite):", (res_surface_AL_SAT.slope)*365*10)
# print("Estimated temperature change per decade (Model):", (res_surface_AL_MODEL.slope)*365*10)
# print("Estimated bottom temperature change per decade (Model):", (res_bottom_AL_MODEL.slope)*365*10)
# print("Estimated RMSE SAT/MODEL:", rmse)

axs.plot(dates_1, surface_AL_SAT_ts, '-', color='black', linewidth=1.5, label='Surface IBI - ODYSSEA L4')

axs.plot(dates_2, surface_AL_MODEL_ts, '-', color='red', linewidth=1.5, label='Surface GLORYS12V1')

bottom_ax = axs.twinx()
bottom_ax.plot(dates_2, bottom_AL_MODEL_ts, '-', color='xkcd:mango', alpha=0.7, linewidth=1.5, label='Bottom GLORYS12V1')


axs.tick_params(length=10, direction='in')
axs.set_ylim(12, 29)
axs.set_yticks([14, 16, 18, 20, 22, 24, 26])
axs.set_ylabel('Surface temperature [$^\circ$C]')
axs.set_xlim(ts, te)
# axs.set_xticks(['1995', '2000', '2005', '2010', '2015', '2020'])
axs.set_title('b     SoG and Alboran Sea (AL)')

bottom_ax.tick_params(length=10, direction='in')
bottom_ax.set_ylim(13, 14.1)  
bottom_ax.set_yticks([13.1, 13.3, 13.5, 13.7, 13.9])  
bottom_ax.set_ylabel('Bottom temperature [$^\circ$C]')  
bottom_ax.spines['right'].set_color('xkcd:mango')
bottom_ax.yaxis.label.set_color('xkcd:mango')
bottom_ax.tick_params(axis='y', colors='xkcd:mango')  
bottom_ax.yaxis.set_tick_params(labelcolor='xkcd:mango')


# fig.legend(loc=(0.13, 0.73), frameon=False, fontsize=14)
fig.legend(loc=(0.08, 0.77), ncol=3, frameon=False, fontsize=18)
# fig.legend(loc='upper left', frameon=False, fontsize=14)



outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\AL_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





                            ##########
                            ##Canary##
                            ##########

## Timeseries of surface temperature
surface_CAN_SAT_ts = np.nanmean(sst_canarias, axis=(1,2))
surface_CAN_MODEL_ts = np.nanmean(thetao_CAN, axis=(1,2))
bottom_CAN_MODEL_ts = np.nanmean(bottomT_CAN, axis=(1,2))

#Mask Satellite data prior to 1993 and/or instrument errors
surface_CAN_SAT_ts[0:4018] = np.NaN


##Remove NaN values
# surface_CAN_SAT_ts = surface_CAN_SAT_ts[~np.isnan(surface_CAN_SAT_ts)]
##Remove outliers: outlier > 3* std
# mean = np.mean(surface_CAN_SAT_ts)
# std = np.std(surface_CAN_SAT_ts)
# threshold = 3 * std
# surface_CAN_SAT_ts = surface_CAN_SAT_ts[abs(surface_CAN_SAT_ts - mean) < threshold]


##Standard error surface temperature
# error_surface_CAN_SAT = (np.nanstd(sst_CAN, axis=(1,2)))/np.sqrt(14975)
# error_surface_CAN_MODEL = (np.nanstd(thetao_CAN, axis=(1,2)))/np.sqrt(10957)


#Times from 01-01-1982 to 31-12-2022 and from 01-01-1993 to 31-12-2022
t, dates_1, T, year, month, day, doy = ecj.timevector([1982, 1, 1], [2022, 12, 31])
t, dates_2, T, year, month, day, doy = ecj.timevector([1993, 1, 1], [2022, 12, 31])

ts = date(1992,7,31)
te = date(2023,7,31)



fig, axs = plt.subplots(figsize=(20, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})

# time_1 = np.arange(1, len(surface_CAN_SAT_ts)+1)
# time_2 = np.arange(1, len(surface_CAN_MODEL_ts)+1)

# #Surface temperature rating change per decade
# res_surface_CAN_SAT = stats.linregress(time_1, surface_CAN_SAT_ts)
# res_surface_CAN_MODEL = stats.linregress(time_2, surface_CAN_MODEL_ts)
# res_bottom_CAN_MODEL = stats.linregress(time_2, bottom_CAN_MODEL_ts)

# Calculate the root mean square error (RMSE) between the satellite and model time series
# Find the index corresponding to the start of the overlapping period
# start_index = np.where(np.array(dates_1) >= datetime.date(1993, 1, 1))[0][0]
# # Trim the satellite time series to match the length of the model time series
# trimmed_surface_CAN_SAT_ts = surface_CAN_SAT_ts[start_index:]
# rmse = np.sqrt(np.mean((trimmed_surface_CAN_SAT_ts - surface_CAN_MODEL_ts) ** 2))

# print("Estimated temperature change per decade (Satellite):", (res_surface_CAN_SAT.slope)*365*10)
# print("Estimated temperature change per decade (Model):", (res_surface_CAN_MODEL.slope)*365*10)
# print("Estimated bottom temperature change per decade (Model):", (res_bottom_CAN_MODEL.slope)*365*10)
# print("Estimated RMSE SAT/MODEL:", rmse)

axs.plot(dates_1, surface_CAN_SAT_ts, '-', color='black', linewidth=1.5, label='Surface OSTIA L4')

axs.plot(dates_2, surface_CAN_MODEL_ts, '-', color='red', linewidth=1.5, label='Surface GLORYS12V1')

bottom_ax = axs.twinx()
bottom_ax.plot(dates_2, bottom_CAN_MODEL_ts, '-', color='xkcd:mango', alpha=0.7, linewidth=1.5, label='Bottom GLORYS12V1')


axs.tick_params(length=10, direction='in')
axs.set_ylim(16.5, 27)
axs.set_yticks([18, 20, 22, 24, 26])
axs.set_ylabel('Surface temperature [$^\circ$C]')
axs.set_xlim(ts, te)
# axs.set_xticks(['1995', '2000', '2005', '2010', '2015', '2020'])
axs.set_title('c     Canary (CAN)')

bottom_ax.tick_params(length=10, direction='in')
bottom_ax.set_ylim(2.78, 3.5)  
bottom_ax.set_yticks([2.8, 2.9, 3, 3.1])  
bottom_ax.set_ylabel('Bottom temperature [$^\circ$C]')  
bottom_ax.spines['right'].set_color('xkcd:mango')
bottom_ax.yaxis.label.set_color('xkcd:mango')
bottom_ax.tick_params(axis='y', colors='xkcd:mango')  
bottom_ax.yaxis.set_tick_params(labelcolor='xkcd:mango')


# fig.legend(loc=(0.13, 0.73), frameon=False, fontsize=14)
fig.legend(loc=(0.08, 0.77), ncol=3, frameon=False, fontsize=18)
# fig.legend(loc='upper left', frameon=False, fontsize=14)



outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\CAN_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)



                            ####################
                            ## South Atlantic ##
                            ####################


## Timeseries of surface temperature
surface_GC_SAT_ts = np.nanmean(sst_GC, axis=(1,2))
surface_GC_MODEL_ts = np.nanmean(thetao_GC, axis=(1,2))
bottom_GC_MODEL_ts = np.nanmean(bottomT_GC, axis=(1,2))

#Mask Satellite data prior to 1993 and/or instrument errors
surface_GC_SAT_ts[0:4018] = np.NaN
surface_GC_SAT_ts[13514:13878] = np.NaN

##Remove NaN values
# surface_GC_SAT_ts = surface_GC_SAT_ts[~np.isnan(surface_GC_SAT_ts)]
##Remove outliers: outlier > 3* std
# mean = np.mean(surface_GC_SAT_ts)
# std = np.std(surface_GC_SAT_ts)
# threshold = 3 * std
# surface_GC_SAT_ts = surface_GC_SAT_ts[abs(surface_GC_SAT_ts - mean) < threshold]


##Standard error surface temperature
# error_surface_GC_SAT = (np.nanstd(sst_GC, axis=(1,2)))/np.sqrt(14975)
# error_surface_GC_MODEL = (np.nanstd(thetao_GC, axis=(1,2)))/np.sqrt(10957)


#Times from 01-01-1982 to 31-12-2022 and from 01-01-1993 to 31-12-2022
t, dates_1, T, year, month, day, doy = ecj.timevector([1982, 1, 1], [2022, 12, 31])
t, dates_2, T, year, month, day, doy = ecj.timevector([1993, 1, 1], [2022, 12, 31])

ts = date(1992,7,31)
te = date(2023,7,31)



fig, axs = plt.subplots(figsize=(20, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})

# time_1 = np.arange(1, len(surface_GC_SAT_ts)+1)
time_2 = np.arange(1, len(surface_GC_MODEL_ts)+1)

# #Surface temperature rating change per decade
# res_surface_GC_SAT = stats.linregress(time_1, surface_GC_SAT_ts)
# res_surface_GC_MODEL = stats.linregress(time_2, surface_GC_MODEL_ts)
res_bottom_GC_MODEL = stats.linregress(time_2, bottom_GC_MODEL_ts)

# Calculate the root mean square error (RMSE) between the satellite and model time series
# Find the index corresponding to the start of the overlapping period
# start_index = np.where(np.array(dates_1) >= datetime.date(1993, 1, 1))[0][0]
# # Trim the satellite time series to match the length of the model time series
# trimmed_surface_GC_SAT_ts = surface_GC_SAT_ts[start_index:]
# rmse = np.sqrt(np.mean((trimmed_surface_GC_SAT_ts - surface_GC_MODEL_ts) ** 2))

# print("Estimated temperature change per decade (Satellite):", (res_surface_GC_SAT.slope)*365*10)
# print("Estimated temperature change per decade (Model):", (res_surface_GC_MODEL.slope)*365*10)
print("Estimated bottom temperature change per decade (Model):", (res_bottom_GC_MODEL.slope)*365*10)
# print("Estimated RMSE SAT/MODEL:", rmse)

axs.plot(dates_1, surface_GC_SAT_ts, '-', color='black', linewidth=1.5, label='Surface IBI - ODYSSEA L4')

axs.plot(dates_2, surface_GC_MODEL_ts, '-', color='red', linewidth=1.5, label='Surface GLORYS12V1')

bottom_ax = axs.twinx()
bottom_ax.plot(dates_2, bottom_GC_MODEL_ts, '-', color='xkcd:mango', alpha=0.7, linewidth=1.5, label='Bottom GLORYS12V1')


axs.tick_params(length=10, direction='in')
axs.set_ylim(13, 27)
axs.set_yticks([16, 18, 20, 22, 24, 26])
axs.set_ylabel('Surface temperature [$^\circ$C]')
axs.set_xlim(ts, te)
# axs.set_xticks(['1995', '2000', '2005', '2010', '2015', '2020'])
axs.set_title('d     South Atlantic (SA)')

bottom_ax.tick_params(length=10, direction='in')
bottom_ax.set_ylim(13.2, 20)  
bottom_ax.set_yticks([13.5, 14.5, 15.5])  
bottom_ax.set_ylabel('Bottom temperature [$^\circ$C]')  
bottom_ax.spines['right'].set_color('xkcd:mango')
bottom_ax.yaxis.label.set_color('xkcd:mango')
bottom_ax.tick_params(axis='y', colors='xkcd:mango')  
bottom_ax.yaxis.set_tick_params(labelcolor='xkcd:mango')


# fig.legend(loc=(0.13, 0.73), frameon=False, fontsize=14)
fig.legend(loc=(0.08, 0.77), ncol=3, frameon=False, fontsize=18)
# fig.legend(loc='upper left', frameon=False, fontsize=14)



outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\SA_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




                            ########################
                            ## Levantine-Balearic ##
                            ########################



## Timeseries of surface temperature
surface_BAL_SAT_ts = np.nanmean(sst_BAL, axis=(1,2))
surface_BAL_MODEL_ts = np.nanmean(thetao_BAL, axis=(1,2))
bottom_BAL_MODEL_ts = np.nanmean(bottomT_BAL, axis=(1,2))

#Mask Satellite data prior to 1993 and/or instrument errors
surface_BAL_SAT_ts[0:4018] = np.NaN
surface_BAL_SAT_ts[13514:13878] = np.NaN

##Remove NaN values
# surface_BAL_SAT_ts = surface_BAL_SAT_ts[~np.isnan(surface_BAL_SAT_ts)]
##Remove outliers: outlier > 3* std
# mean = np.mean(surface_BAL_SAT_ts)
# std = np.std(surface_BAL_SAT_ts)
# threshold = 3 * std
# surface_BAL_SAT_ts = surface_BAL_SAT_ts[abs(surface_BAL_SAT_ts - mean) < threshold]


##Standard error surface temperature
# error_surface_BAL_SAT = (np.nanstd(sst_BAL, axis=(1,2)))/np.sqrt(14975)
# error_surface_BAL_MODEL = (np.nanstd(thetao_BAL, axis=(1,2)))/np.sqrt(10957)


#Times from 01-01-1982 to 31-12-2022 and from 01-01-1993 to 31-12-2022
t, dates_1, T, year, month, day, doy = ecj.timevector([1982, 1, 1], [2022, 12, 31])
t, dates_2, T, year, month, day, doy = ecj.timevector([1993, 1, 1], [2022, 12, 31])

ts = date(1992,7,31)
te = date(2023,7,31)



fig, axs = plt.subplots(figsize=(20, 5))
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
# plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.size': 20, 'font.family': 'Arial'})

# time_1 = np.arange(1, len(surface_BAL_SAT_ts)+1)
time_2 = np.arange(1, len(surface_BAL_MODEL_ts)+1)

# #Surface temperature rating change per decade
# res_surface_BAL_SAT = stats.linregress(time_1, surface_BAL_SAT_ts)
# res_surface_BAL_MODEL = stats.linregress(time_2, surface_BAL_MODEL_ts)
res_bottom_BAL_MODEL = stats.linregress(time_2, bottom_BAL_MODEL_ts)

# Calculate the root mean square error (RMSE) between the satellite and model time series
# Find the index corresponding to the start of the overlapping period
# start_index = np.where(np.array(dates_1) >= datetime.date(1993, 1, 1))[0][0]
# # Trim the satellite time series to match the length of the model time series
# trimmed_surface_BAL_SAT_ts = surface_BAL_SAT_ts[start_index:]
# rmse = np.sqrt(np.mean((trimmed_surface_BAL_SAT_ts - surface_BAL_MODEL_ts) ** 2))

# print("Estimated temperature change per decade (Satellite):", (res_surface_BAL_SAT.slope)*365*10)
# print("Estimated temperature change per decade (Model):", (res_surface_BAL_MODEL.slope)*365*10)
print("Estimated bottom temperature change per decade (Model):", (res_bottom_BAL_MODEL.slope)*365*10)
# print("Estimated RMSE SAT/MODEL:", rmse)

axs.plot(dates_1, surface_BAL_SAT_ts, '-', color='black', linewidth=1.5, label='Surface IBI - ODYSSEA L4')

axs.plot(dates_2, surface_BAL_MODEL_ts, '-', color='red', linewidth=1.5, label='Surface GLORYS12V1')

bottom_ax = axs.twinx()
bottom_ax.plot(dates_2, bottom_BAL_MODEL_ts, '-', color='xkcd:mango', alpha=0.7, linewidth=1.5, label='Bottom GLORYS12V1')


axs.tick_params(length=10, direction='in')
axs.set_ylim(12, 32)
axs.set_yticks([14, 16, 18, 20, 22, 24, 26, 28])
axs.set_ylabel('Surface temperature [$^\circ$C]')
axs.set_xlim(ts, te)
# axs.set_xticks(['1995', '2000', '2005', '2010', '2015', '2020'])
axs.set_title('e     Levantine-Balearic (BAL)')

bottom_ax.tick_params(length=10, direction='in')
bottom_ax.set_ylim(12.9, 14.7)  
bottom_ax.set_yticks([13, 13.2, 13.4, 13.6])  
bottom_ax.set_ylabel('Bottom temperature [$^\circ$C]')  
bottom_ax.spines['right'].set_color('xkcd:mango')
bottom_ax.yaxis.label.set_color('xkcd:mango')
bottom_ax.tick_params(axis='y', colors='xkcd:mango')  
bottom_ax.yaxis.set_tick_params(labelcolor='xkcd:mango')


# fig.legend(loc=(0.13, 0.73), frameon=False, fontsize=14)
fig.legend(loc=(0.08, 0.77), ncol=3, frameon=False, fontsize=18)
# fig.legend(loc='upper left', frameon=False, fontsize=14)



outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\BAL_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)








                        



