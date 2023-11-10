# -*- coding: utf-8 -*-
"""

######################### Climatologies Comparison SAT/MODEL ##################

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
import seaborn as sns

import datetime
from datetime import date
import ecoliver as ecj


##Load MHWs_from_MATLAB.py##




                    ################################
                    ### CLIMATOLOGIES COMPARISON ###
                    ################################

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

Td_GC_SAT_1_ts = np.nanmean(np.where(MHW_td_ts_GC_SAT_1 == 0, np.NaN, MHW_td_ts_GC_SAT_1), axis=(0,1))
Td_GC_SAT_ts = np.nanmean(np.where(MHW_td_ts_GC_SAT == 0, np.NaN, MHW_td_ts_GC_SAT), axis=(0,1)) 
Td_GC_MODEL_ts = np.nanmean(np.where(MHW_td_ts_GC_MODEL == 0, np.NaN, MHW_td_ts_GC_MODEL), axis=(0,1))
BTd_GC_MODEL_ts = np.nanmean(np.where(BMHW_td_ts_GC_MODEL == 0, np.NaN, BMHW_td_ts_GC_MODEL), axis=(0,1))

Td_BAL_SAT_1_ts = np.nanmean(np.where(MHW_td_ts_BAL_SAT_1 == 0, np.NaN, MHW_td_ts_BAL_SAT_1), axis=(0,1))
Td_BAL_SAT_ts = np.nanmean(np.where(MHW_td_ts_BAL_SAT == 0, np.NaN, MHW_td_ts_BAL_SAT), axis=(0,1)) 
Td_BAL_MODEL_ts = np.nanmean(np.where(MHW_td_ts_BAL_MODEL == 0, np.NaN, MHW_td_ts_BAL_MODEL), axis=(0,1))
BTd_BAL_MODEL_ts = np.nanmean(np.where(BMHW_td_ts_BAL_MODEL == 0, np.NaN, BMHW_td_ts_BAL_MODEL), axis=(0,1))


# Cum_NA_SAT_1_ts = np.nanmean(MHW_cum_ts_NA_SAT_1, axis=(0,1))
# Cum_NA_SAT_ts = np.nanmean(MHW_cum_ts_NA_SAT, axis=(0,1)) 
# Cum_NA_MODEL_ts = np.nanmean(MHW_cum_ts_NA_MODEL, axis=(0,1))                           

# Cum_AL_SAT_1_ts = np.nanmean(MHW_cum_ts_AL_SAT_1, axis=(0,1))
# Cum_AL_SAT_ts = np.nanmean(MHW_cum_ts_AL_SAT, axis=(0,1)) 
# Cum_AL_MODEL_ts = np.nanmean(MHW_cum_ts_AL_MODEL, axis=(0,1))    

# Cum_CAN_SAT_1_ts = np.nanmean(MHW_cum_ts_CAN_SAT_1, axis=(0,1))
# Cum_CAN_SAT_ts = np.nanmean(MHW_cum_ts_CAN_SAT, axis=(0,1)) 
# Cum_CAN_MODEL_ts = np.nanmean(MHW_cum_ts_CAN_MODEL, axis=(0,1))      

# Cum_GC_SAT_1_ts = np.nanmean(MHW_cum_ts_GC_SAT_1, axis=(0,1))
# Cum_GC_SAT_ts = np.nanmean(MHW_cum_ts_GC_SAT, axis=(0,1)) 
# Cum_GC_MODEL_ts = np.nanmean(MHW_cum_ts_GC_MODEL, axis=(0,1)) 

# Cum_BAL_SAT_1_ts = np.nanmean(MHW_cum_ts_BAL_SAT_1, axis=(0,1))
# Cum_BAL_SAT_ts = np.nanmean(MHW_cum_ts_BAL_SAT, axis=(0,1)) 
# Cum_BAL_MODEL_ts = np.nanmean(MHW_cum_ts_BAL_MODEL, axis=(0,1)) 


# Max_NA_SAT_1_ts = np.nanmean(MHW_max_ts_NA_SAT_1, axis=(0,1))
# Max_NA_SAT_ts = np.nanmean(MHW_max_ts_NA_SAT, axis=(0,1)) 
# Max_NA_MODEL_ts = np.nanmean(MHW_max_ts_NA_MODEL, axis=(0,1))                           

# Max_AL_SAT_1_ts = np.nanmean(MHW_max_ts_AL_SAT_1, axis=(0,1))
# Max_AL_SAT_ts = np.nanmean(MHW_max_ts_AL_SAT, axis=(0,1)) 
# Max_AL_MODEL_ts = np.nanmean(MHW_max_ts_AL_MODEL, axis=(0,1))    

# Max_CAN_SAT_1_ts = np.nanmean(MHW_max_ts_CAN_SAT_1, axis=(0,1))
# Max_CAN_SAT_ts = np.nanmean(MHW_max_ts_CAN_SAT, axis=(0,1)) 
# Max_CAN_MODEL_ts = np.nanmean(MHW_max_ts_CAN_MODEL, axis=(0,1))      

# Max_GC_SAT_1_ts = np.nanmean(MHW_max_ts_GC_SAT_1, axis=(0,1))
# Max_GC_SAT_ts = np.nanmean(MHW_max_ts_GC_SAT, axis=(0,1)) 
# Max_GC_MODEL_ts = np.nanmean(MHW_max_ts_GC_MODEL, axis=(0,1)) 

# Max_BAL_SAT_1_ts = np.nanmean(MHW_max_ts_BAL_SAT_1, axis=(0,1))
# Max_BAL_SAT_ts = np.nanmean(MHW_max_ts_BAL_SAT, axis=(0,1)) 
# Max_BAL_MODEL_ts = np.nanmean(MHW_max_ts_BAL_MODEL, axis=(0,1))





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

# sns.despine()

# axs.set_xlabel(r'Spatially Averaged SMHW Maximum Intensity [$^\circ$C]', fontsize=15)
axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
# axs.set_xlabel(r'Spatially Averaged SMHW Cumulative Intensity [$^\circ$C · days]', fontsize=15)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'North Atlantic (NA)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.066)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)
legend = plt.legend(fontsize=14, loc='best', frameon=False)

plt.tight_layout()


#Save the figure so far
outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\NA_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







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

# sns.despine()

# axs.set_xlabel(r'Spatially Averaged SMHW Maximum Intensity [$^\circ$C]', fontsize=15)
axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
# axs.set_xlabel(r'Spatially Averaged SMHW Cumulative Intensity [$^\circ$C · days]', fontsize=15)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'SoG and Alboran Sea (AL)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.05)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)

plt.tight_layout()



#Save the figure so far
outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\AL_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)





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

# sns.despine()

# axs.set_xlabel(r'Spatially Averaged SMHW Maximum Intensity [$^\circ$C]', fontsize=15)
axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
# axs.set_xlabel(r'Spatially Averaged SMHW Cumulative Intensity [$^\circ$C · days]', fontsize=15)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'Canary (CAN)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.05)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)
# legend = plt.legend(['OSTIA L4 [ref: 1982-2012]', 'OSTIA L4 [ref: 1993-2022]', '_', '_'],fontsize=15)

plt.tight_layout()


#Save the figure so far
outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\CAN_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




                        ####################
                        ## South Atlantic ##
                        ####################
                                
#Create a DataFrame for the data using pandas
data = pd.DataFrame({
    'IBI - ODYSSEA L4 [reference: 1982-2012]': Td_GC_SAT_1_ts[11:41],
    'IBI - ODYSSEA L4 [reference: 1993-2022]': Td_GC_SAT_ts,
    'Surface GLORYS12V1 [reference: 1993-2022]': Td_GC_MODEL_ts,
    'Bottom GLORYS12V1 [reference: 1993-2022]': BTd_GC_MODEL_ts})


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

# sns.despine()

# axs.set_xlabel(r'Spatially Averaged SMHW Maximum Intensity [$^\circ$C]', fontsize=15)
axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
# axs.set_xlabel(r'Spatially Averaged SMHW Cumulative Intensity [$^\circ$C · days]', fontsize=15)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'South Atlantic (SA)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.05)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)

plt.tight_layout()


#Save the figure so far
outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\SA_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)




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

# sns.despine()

# axs.set_xlabel(r'Spatially Averaged SMHW Maximum Intensity [$^\circ$C]', fontsize=15)
axs.set_xlabel(r'Total Annual MHW Days [days]', fontsize=22)
# axs.set_xlabel(r'Spatially Averaged SMHW Cumulative Intensity [$^\circ$C · days]', fontsize=15)
axs.set_ylabel(r'Density', fontsize=22)
axs.set_title(r'Levantine-Balearic (BAL)', fontsize=26)
axs.set_xlim(-50, 200)
axs.set_ylim(0, 0.05)
plt.xticks(fontsize=22)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05], fontsize=22)

plt.tight_layout()


#Save the figure so far
outfile = r'C:\Users\Manuel\Desktop\Paper_EEMM_MHWs\Figuras_Paper_EEMM\Fig_2\BAL_Clim_Comparison_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight', pad_inches=0.5)







