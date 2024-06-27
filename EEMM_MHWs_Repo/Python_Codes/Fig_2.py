# -*- coding: utf-8 -*-
"""

#########################      Figure 2 in 
Fernández-Barba, M., Huertas, I. E., & Navarro, G. (2024). 
Assessment of surface and bottom marine heatwaves along the Spanish coast. 
Ocean Modelling, 190, 102399.                          ########################

"""

#Loading required python modules
import numpy as np
import xarray as xr 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from datetime import date
import ecoliver as ecj


##Load previously-clipped temperature datasets (Datos_ESMARES.py)
##Load MHWs_from_MATLAB.py


#North Atlantic
ds_NA = xr.open_dataset(r'.../SST_NA_clipped.nc')
sst_NA = ds_NA['analysed_sst'] - 273.15 #K to ºC
ds_Model_NA = xr.open_dataset(r'...\T_GLORYS_NA_clipped.nc')
thetao_NA = ds_Model_NA['thetao']
ds_Model_NA = xr.open_dataset(r'...\bottomT_GLORYS_NA_clipped.nc')
bottomT_NA = ds_Model_NA['bottomT']

#SoG and Alboran
ds_AL = xr.open_dataset(r'.../SST_AL_clipped.nc')
sst_AL = ds_AL['analysed_sst'] - 273.15 #K to ºC
ds_Model_AL = xr.open_dataset(r'...\Temp_MLT_GLORYS_AL_clipped.nc')
thetao_AL = ds_Model_AL['thetao']
bottomT_AL = ds_Model_AL['bottomT']

#Canary
ds_canarias = xr.open_dataset(r'.../SST_Canary_clipped.nc')
sst_canarias = ds_canarias['analysed_sst'] - 273.15 #K to ºC
ds_Model_CAN = xr.open_dataset(r'...\Temp_MLT_GLORYS_Canary_clipped.nc')
thetao_CAN = ds_Model_CAN['thetao']
bottomT_CAN = ds_Model_CAN['bottomT']

#South Atlantic
ds_SA = xr.open_dataset(r'.../SST_SA_clipped.nc')
sst_SA = ds_SA['analysed_sst'] - 273.15 #K to ºC
ds_Model_SA = xr.open_dataset(r'...\Temp_MLT_GLORYS_SA_clipped.nc')
thetao_SA = ds_Model_SA['thetao']
bottomT_SA = ds_Model_SA['bottomT']

#Levantine-Balearic
ds_BAL = xr.open_dataset(r'.../SST_BAL_clipped.nc')
sst_BAL = ds_BAL['analysed_sst'] - 273.15 #K to ºC
ds_Model_BAL = xr.open_dataset(r'...\T_GLORYS_BAL_clipped.nc')
thetao_BAL = ds_Model_BAL['thetao']
ds_Model_BAL = xr.open_dataset(r'...\bottomT_GLORYS_BAL_clipped.nc')
bottomT_BAL = ds_Model_BAL['bottomT']




##Figs 2a, c, e, g, i (Surface and bottom temperatures comparison)

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


##Standard errors
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



outfile = r'...\Fig_2\2a_NA_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')





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


outfile = r'...\Fig_2\2c_AL_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')





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


outfile = r'...\Fig_2\2e_CAN_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')



                            ####################
                            ## South Atlantic ##
                            ####################


## Timeseries of surface temperature
surface_SA_SAT_ts = np.nanmean(sst_SA, axis=(1,2))
surface_SA_MODEL_ts = np.nanmean(thetao_SA, axis=(1,2))
bottom_SA_MODEL_ts = np.nanmean(bottomT_SA, axis=(1,2))

#Mask Satellite data prior to 1993 and/or instrument errors
surface_SA_SAT_ts[0:4018] = np.NaN
surface_SA_SAT_ts[13514:13878] = np.NaN

##Remove NaN values
# surface_SA_SAT_ts = surface_SA_SAT_ts[~np.isnan(surface_SA_SAT_ts)]
##Remove outliers: outlier > 3* std
# mean = np.mean(surface_SA_SAT_ts)
# std = np.std(surface_SA_SAT_ts)
# threshold = 3 * std
# surface_SA_SAT_ts = surface_SA_SAT_ts[abs(surface_SA_SAT_ts - mean) < threshold]


##Standard error surface temperature
# error_surface_SA_SAT = (np.nanstd(sst_SA, axis=(1,2)))/np.sqrt(14975)
# error_surface_SA_MODEL = (np.nanstd(thetao_SA, axis=(1,2)))/np.sqrt(10957)


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

# time_1 = np.arange(1, len(surface_SA_SAT_ts)+1)
time_2 = np.arange(1, len(surface_SA_MODEL_ts)+1)

# #Surface temperature rating change per decade
# res_surface_SA_SAT = stats.linregress(time_1, surface_SA_SAT_ts)
# res_surface_SA_MODEL = stats.linregress(time_2, surface_SA_MODEL_ts)
res_bottom_SA_MODEL = stats.linregress(time_2, bottom_SA_MODEL_ts)

# Calculate the root mean square error (RMSE) between the satellite and model time series
# Find the index corresponding to the start of the overlapping period
# start_index = np.where(np.array(dates_1) >= datetime.date(1993, 1, 1))[0][0]
# # Trim the satellite time series to match the length of the model time series
# trimmed_surface_SA_SAT_ts = surface_SA_SAT_ts[start_index:]
# rmse = np.sqrt(np.mean((trimmed_surface_SA_SAT_ts - surface_SA_MODEL_ts) ** 2))

# print("Estimated temperature change per decade (Satellite):", (res_surface_SA_SAT.slope)*365*10)
# print("Estimated temperature change per decade (Model):", (res_surface_SA_MODEL.slope)*365*10)
print("Estimated bottom temperature change per decade (Model):", (res_bottom_SA_MODEL.slope)*365*10)
# print("Estimated RMSE SAT/MODEL:", rmse)

axs.plot(dates_1, surface_SA_SAT_ts, '-', color='black', linewidth=1.5, label='Surface IBI - ODYSSEA L4')

axs.plot(dates_2, surface_SA_MODEL_ts, '-', color='red', linewidth=1.5, label='Surface GLORYS12V1')

bottom_ax = axs.twinx()
bottom_ax.plot(dates_2, bottom_SA_MODEL_ts, '-', color='xkcd:mango', alpha=0.7, linewidth=1.5, label='Bottom GLORYS12V1')


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


outfile = r'...\Fig_2\2g_SA_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




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


outfile = r'...\Fig_2\2i_BAL_SAT_MODEL.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




##Figs 2b, d, f, h, j (Surface and bottom Total Annual MHWs comparison)


#Time arrays to represent MHW metrics
time_1 = np.arange(1982, 2023)
time = np.arange(1993, 2023)

#Spatially averaged metrics         
Td_NA_SAT_1_ts = np.nanmean(np.where(MHW_td_ts_NA_SAT_1 == 0, np.NaN, MHW_td_ts_NA_SAT_1), axis=(0,1))
Td_NA_SAT_ts = np.nanmean(np.where(MHW_td_ts_NA_SAT == 0, np.NaN, MHW_td_ts_NA_SAT), axis=(0,1)) 
Td_NA_MODEL_ts = np.nanmean(np.where(MHW_td_ts_NA_MODEL == 0, np.NaN, MHW_td_ts_NA_MODEL), axis=(0,1))
BTd_NA_MODEL_ts = np.nanmean(np.where(BMHW_td_ts_NA_MODEL == 0, np.NaN, BMHW_td_ts_NA_MODEL), axis=(0,1))

Td_AL_SAT_1_ts = np.nanmean(np.where(MHW_td_ts_AL_SAT_1 == 0, np.NaN, MHW_td_ts_AL_SAT_1), axis=(0,1))
Td_AL_SAT_ts = np.nanmean(np.where(MHW_td_ts_AL_SAT == 0, np.NaN, MHW_td_ts_AL_SAT), axis=(0,1)) 
Td_AL_MODEL_ts = np.nanmean(np.where(MHW_td_ts_AL_MODEL == 0, np.NaN, MHW_td_ts_AL_MODEL), axis=(0,1))
BTd_AL_MODEL_ts = np.nanmean(np.where(BMHW_td_ts_AL_MODEL == 0, np.NaN, BMHW_td_ts_AL_MODEL), axis=(0,1))

Td_CAN_SAT_1_ts = np.nanmean(np.where(MHW_td_ts_CAN_SAT_1 == 0, np.NaN, MHW_td_ts_CAN_SAT_1), axis=(0,1))
Td_CAN_SAT_ts = np.nanmean(np.where(MHW_td_ts_CAN_SAT == 0, np.NaN, MHW_td_ts_CAN_SAT), axis=(0,1)) 
Td_CAN_MODEL_ts = np.nanmean(np.where(MHW_td_ts_CAN_MODEL == 0, np.NaN, MHW_td_ts_CAN_MODEL), axis=(0,1))
BTd_CAN_MODEL_ts = np.nanmean(np.where(BMHW_td_ts_CAN_MODEL == 0, np.NaN, BMHW_td_ts_CAN_MODEL), axis=(0,1))

Td_SA_SAT_1_ts = np.nanmean(np.where(MHW_td_ts_SA_SAT_1 == 0, np.NaN, MHW_td_ts_SA_SAT_1), axis=(0,1))
Td_SA_SAT_ts = np.nanmean(np.where(MHW_td_ts_SA_SAT == 0, np.NaN, MHW_td_ts_SA_SAT), axis=(0,1)) 
Td_SA_MODEL_ts = np.nanmean(np.where(MHW_td_ts_SA_MODEL == 0, np.NaN, MHW_td_ts_SA_MODEL), axis=(0,1))
BTd_SA_MODEL_ts = np.nanmean(np.where(BMHW_td_ts_SA_MODEL == 0, np.NaN, BMHW_td_ts_SA_MODEL), axis=(0,1))

Td_BAL_SAT_1_ts = np.nanmean(np.where(MHW_td_ts_BAL_SAT_1 == 0, np.NaN, MHW_td_ts_BAL_SAT_1), axis=(0,1))
Td_BAL_SAT_ts = np.nanmean(np.where(MHW_td_ts_BAL_SAT == 0, np.NaN, MHW_td_ts_BAL_SAT), axis=(0,1)) 
Td_BAL_MODEL_ts = np.nanmean(np.where(MHW_td_ts_BAL_MODEL == 0, np.NaN, MHW_td_ts_BAL_MODEL), axis=(0,1))
BTd_BAL_MODEL_ts = np.nanmean(np.where(BMHW_td_ts_BAL_MODEL == 0, np.NaN, BMHW_td_ts_BAL_MODEL), axis=(0,1))



##Representing density function of the total annual MHWs days along the five spanish demarcations

                        ####################
                        ## North Atlantic ##
                        ####################

#Create a DataFrame for the data using pandas
data = pd.DataFrame({
    'IBI - ODYSSEA L4 [reference: 1982-2012]': Td_NA_SAT_1_ts[11:41],
    'IBI - ODYSSEA L4 [reference: 1993-2022]': Td_NA_SAT_ts,
    'Surface GLORYS12V1 [reference: 1993-2022]': Td_NA_MODEL_ts,
    'Bottom GLORYS12V1 [reference: 1993-2022]': BTd_NA_MODEL_ts})



#Configure the style of the plots
sns.set(style="ticks")

# Set custom colors for each series
colors = ['black', 'black', 'red', 'gold']

# Kernel Density Estimation (KDE) plot
fig, axs = plt.subplots(figsize=(6, 5))
sns.set_context("paper")
sns.set_palette(colors)

sns.kdeplot(data=data['IBI - ODYSSEA L4 [reference: 1982-2012]'], fill=True, color='none', edgecolor='black', common_norm=False, label='IBI - ODYSSEA L4 [ref: 1982-2012]', linewidth=2, hatch='//')
sns.kdeplot(data=data['IBI - ODYSSEA L4 [reference: 1993-2022]'], fill=True, color='black', common_norm=False, label='IBI - ODYSSEA L4 [ref: 1993-2022]', linewidth=2)
sns.kdeplot(data=data['Surface GLORYS12V1 [reference: 1993-2022]'], fill=True, color='red', common_norm=False, label='Surface GLORYS12V1 [ref: 1993-2022]', linewidth=2)
sns.kdeplot(data=data['Bottom GLORYS12V1 [reference: 1993-2022]'], fill=True, color='gold', common_norm=False, label='Bottom GLORYS12V1 [ref: 1993-2022]', linewidth=2)

axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'North Atlantic (NA)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.066)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)
legend = plt.legend(fontsize=14, loc='best', frameon=False)

plt.tight_layout()


#Save the figure so far
outfile = r'...\Fig_2\2b_NA_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




                        #########################
                        ## SoG and Alboran Sea ##
                        #########################
                                
#Create a DataFrame for the data using pandas
data = pd.DataFrame({
    'IBI - ODYSSEA L4 [reference: 1982-2012]': Td_AL_SAT_1_ts[11:41],
    'IBI - ODYSSEA L4 [reference: 1993-2022]': Td_AL_SAT_ts,
    'Surface GLORYS12V1 [reference: 1993-2022]': Td_AL_MODEL_ts,
    'Bottom GLORYS12V1 [reference: 1993-2022]': BTd_AL_MODEL_ts})


#Configure the style of the plots
sns.set(style="ticks")

# Set custom colors for each series
colors = ['darkblue', 'black', 'red', 'gold']


# Kernel Density Estimation (KDE) plot
fig, axs = plt.subplots(figsize=(6, 5))
sns.set_context("paper")
sns.set_palette(colors)

sns.kdeplot(data=data['IBI - ODYSSEA L4 [reference: 1982-2012]'], fill=True, color='none', edgecolor='black', common_norm=False, label='IBI - ODYSSEA L4 [ref: 1982-2012]', linewidth=2, hatch='//')
sns.kdeplot(data=data['IBI - ODYSSEA L4 [reference: 1993-2022]'], fill=True, color='black', common_norm=False, label='IBI - ODYSSEA L4 [ref: 1993-2022]', linewidth=2)
sns.kdeplot(data=data['Surface GLORYS12V1 [reference: 1993-2022]'], fill=True, color='red', common_norm=False, label='Surface GLORYS12V1 [ref: 1993-2022]', linewidth=2)
sns.kdeplot(data=data['Bottom GLORYS12V1 [reference: 1993-2022]'], fill=True, color='gold', common_norm=False, label='Bottom GLORYS12V1 [ref: 1993-2022]', linewidth=2)


axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'SoG and Alboran Sea (AL)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.05)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)

plt.tight_layout()


#Save the figure so far
outfile = r'...\Fig_2\2d_AL_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')





                            ############
                            ## Canary ##
                            ############
                                
#Create a DataFrame for the data using pandas
data = pd.DataFrame({
    'OSTIA L4 [reference: 1982-2012]': Td_CAN_SAT_1_ts[11:41],
    'OSTIA L4 [reference: 1993-2022]': Td_CAN_SAT_ts,
    'Surface GLORYS12V1 [reference: 1993-2022]': Td_CAN_MODEL_ts,
    'Bottom GLORYS12V1 [reference: 1993-2022]': BTd_CAN_MODEL_ts})


#Configure the style of the plots
sns.set(style="ticks")

# Set custom colors for each series
colors = ['darkblue', 'black', 'red', 'gold']


# Kernel Density Estimation (KDE) plot
fig, axs = plt.subplots(figsize=(6, 5))
sns.set_context("paper")
sns.set_palette(colors)

sns.kdeplot(data=data['OSTIA L4 [reference: 1982-2012]'], fill=True, color='none', edgecolor='black', common_norm=False, label=False, linewidth=2, hatch='//')
sns.kdeplot(data=data['OSTIA L4 [reference: 1993-2022]'], fill=True, color='black', common_norm=False, label=False, linewidth=2)
sns.kdeplot(data=data['Surface GLORYS12V1 [reference: 1993-2022]'], fill=True, color='red', common_norm=False, label=False, linewidth=2)
sns.kdeplot(data=data['Bottom GLORYS12V1 [reference: 1993-2022]'], fill=True, color='gold', common_norm=False, label=False, linewidth=2)


axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'Canary (CAN)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.05)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)

plt.tight_layout()


#Save the figure so far
outfile = r'...\Fig_2\2f_CAN_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




                        ####################
                        ## South Atlantic ##
                        ####################
                                
#Create a DataFrame for the data using pandas
data = pd.DataFrame({
    'IBI - ODYSSEA L4 [reference: 1982-2012]': Td_SA_SAT_1_ts[11:41],
    'IBI - ODYSSEA L4 [reference: 1993-2022]': Td_SA_SAT_ts,
    'Surface GLORYS12V1 [reference: 1993-2022]': Td_SA_MODEL_ts,
    'Bottom GLORYS12V1 [reference: 1993-2022]': BTd_SA_MODEL_ts})


#Configure the style of the plots
sns.set(style="ticks")

# Set custom colors for each series
colors = ['darkblue', 'black', 'red', 'gold']


# Kernel Density Estimation (KDE) plot
fig, axs = plt.subplots(figsize=(6, 5))
sns.set_context("paper")
sns.set_palette(colors)

sns.kdeplot(data=data['IBI - ODYSSEA L4 [reference: 1982-2012]'], fill=True, color='none', edgecolor='black', common_norm=False, label='IBI - ODYSSEA L4 [ref: 1982-2012]', linewidth=2, hatch='//')
sns.kdeplot(data=data['IBI - ODYSSEA L4 [reference: 1993-2022]'], fill=True, color='black', common_norm=False, label='IBI - ODYSSEA L4 [ref: 1993-2022]', linewidth=2)
sns.kdeplot(data=data['Surface GLORYS12V1 [reference: 1993-2022]'], fill=True, color='red', common_norm=False, label='Surface GLORYS12V1 [ref: 1993-2022]', linewidth=2)
sns.kdeplot(data=data['Bottom GLORYS12V1 [reference: 1993-2022]'], fill=True, color='gold', common_norm=False, label='Bottom GLORYS12V1 [ref: 1993-2022]', linewidth=2)


axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'South Atlantic (SA)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.05)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)

plt.tight_layout()


#Save the figure so far
outfile = r'...\Fig_2\2f_SA_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




                        ########################
                        ## Levantine-Balearic ##
                        ########################
                                
#Create a DataFrame for the data using pandas
data = pd.DataFrame({
    'IBI - ODYSSEA L4 [reference: 1982-2012]': Td_BAL_SAT_1_ts[11:41],
    'IBI - ODYSSEA L4 [reference: 1993-2022]': Td_BAL_SAT_ts,
    'Surface GLORYS12V1 [reference: 1993-2022]': Td_BAL_MODEL_ts,
    'Bottom GLORYS12V1 [reference: 1993-2022]': BTd_BAL_MODEL_ts})

#Configure the style of the plots
sns.set(style="ticks")

# Set custom colors for each series
colors = ['darkblue', 'black', 'red', 'gold']


# Kernel Density Estimation (KDE) plot
fig, axs = plt.subplots(figsize=(6, 5))
sns.set_context("paper")
sns.set_palette(colors)

sns.kdeplot(data=data['IBI - ODYSSEA L4 [reference: 1982-2012]'], fill=True, color='none', edgecolor='black', common_norm=False, label='IBI - ODYSSEA L4 [ref: 1982-2012]', linewidth=2, hatch='//')
sns.kdeplot(data=data['IBI - ODYSSEA L4 [reference: 1993-2022]'], fill=True, color='black', common_norm=False, label='IBI - ODYSSEA L4 [ref: 1993-2022]', linewidth=2)
sns.kdeplot(data=data['Surface GLORYS12V1 [reference: 1993-2022]'], fill=True, color='red', common_norm=False, label='Surface GLORYS12V1 [ref: 1993-2022]', linewidth=2)
sns.kdeplot(data=data['Bottom GLORYS12V1 [reference: 1993-2022]'], fill=True, color='gold', common_norm=False, label='Bottom GLORYS12V1 [ref: 1993-2022]', linewidth=2)


axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'Levantine-Balearic (BAL)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.05)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)

plt.tight_layout()


#Save the figure so far
outfile = r'...\Fig_2\2j_BAL_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')



