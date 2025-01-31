# -*- coding: utf-8 -*-
"""

########################## CHL Phenology metrics  #############################

"""

#Loading required libraries
import numpy as np
import xarray as xr 

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# import cmocean as cm

from scipy import stats
from scipy.ndimage import label

from tqdm import tqdm



##Calculating Phenology metrics

######## Bloom detect #######
def bloom_detect(array, peak_day):
    ini = []
    fin = []

    for i in range(len(array)):
        if array[i] == peak_day:
            # Found the maximum day, now going back to find the beginning of the event
            j = i
            while j > 0 and array[j] == array[j - 1] + 1:
                j -= 1
            ini = array[j]

            # Moving forward to find the end of the event
            j = i
            while j < len(array) - 1 and array[j] == array[j + 1] - 1:
                j += 1
            fin = array[j]

            break  # Found the event, exiting the loop

    return ini, fin

## Example
# array = bloom_indices
# peak_day = peak
# ini, fin = bloom_detect(array, peak_day)
# print(f"Initial bloom day: {ini}. Bloom termination day: {fin}")
#############################



# Loading dataset
ds = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_CAN.nc')

# Lat and Lon matrices for representation             
LAT, LON = np.meshgrid(ds.lat, ds.lon)
LAT, LON = LAT.T, LON.T 


# Apply rolling mean
ds_smoothed = ds.rolling(time=21, center=True, min_periods=1).mean() #3-week rolling mean


# ds.chlor_a[900:963, :, :].mean(dim=('lat','lon'), skipna=True).plot(color='green') 
# ds_smoothed.chlor_a[900:963, :, :].mean(dim=('lat','lon'), skipna=True).plot(color='red') #Testing smoothing


# Calculate annual mean, median and threshold
# annual_mean = ds_smoothed.groupby('time.year').mean(dim='time', skipna=True)
# annual_median = ds_smoothed.groupby('time.year').median(dim='time', skipna=True)
# threshold = annual_median*1.05
# climatology = ds.groupby('time.dayofyear').mean(dim='time')


# Define intervals for each year
last_year = "2023"
year_intervals = [(f"{year}-01-01", f"{year}-12-31") for year in range(1998, 2024)]
# year_intervals[-1] = (f"{last_year}-01-01", f"{last_year}-11-10")  # Adjust last interval when last year is incomplete

# Matrix sizes
lat_size = ds.sizes['lat']
lon_size = ds.sizes['lon']
year_intervals_size = len(year_intervals)


# Initialize arrays for metrics with np.NaN
bloom_freq = np.full((lat_size, lon_size, year_intervals_size), np.NaN)
bloom_max = np.full_like(bloom_freq, np.NaN)
bloom_peak = np.full_like(bloom_freq, np.NaN)
bloom_amp = np.full_like(bloom_freq, np.NaN)
bloom_ini = np.full_like(bloom_freq, np.NaN)
bloom_fin = np.full_like(bloom_freq, np.NaN)
bloom_dur = np.full_like(bloom_freq, np.NaN)
bloom_main_cum = np.full_like(bloom_freq, np.NaN)
bloom_total_cum = np.full_like(bloom_freq, np.NaN)


# Loop over years
for year, (start_date, end_date) in enumerate(tqdm(year_intervals, desc='Processing years')):
    # Filter data for the current year
    # ds_year = ds_smoothed.sel(time=slice(start_date, end_date))
    ds_year = ds.sel(time=slice(start_date, end_date))

    # Loop over grid cells
    for i in tqdm(range(ds_year.sizes['lat']), desc='Processing rows', leave=False):
        for j in tqdm(range(ds_year.sizes['lon']), desc='Processing columns', leave=False):
            
            # Get the time series for the pixel
            pixel_series = ds_year.isel(lat=i, lon=j).CHL.values.squeeze()
            
            
            # Check if all values in the series are NaN
            if not np.all(np.isnan(pixel_series)):
                
                # Get the threshold for the current year (5% of the annual Chl-a median)
                threshold_year = np.nanmedian(pixel_series) * 1.05
                mean_value = np.nanmean(pixel_series)
            
                
                ##CONDITIONS
                # FIRST CONDITION: Find where the Chl-a series exceeds the threshold (5% of the annual Chl-a median)
                bloom_condition = pixel_series > threshold_year
            
                # SECOND CONDITION: First condition must maintained for at least 15 days
                consecutive_true_count = 0

                for idx in range(len(bloom_condition)):
                    if bloom_condition[idx]:
                        consecutive_true_count += 1
                    else:
                        if consecutive_true_count <= 14:
                            # Set to False if there are 14 or fewer consecutive True values
                            bloom_condition[idx - consecutive_true_count:idx] = False
                        consecutive_true_count = 0

                # THIRD CONDITION: 2 days gap between 2 events will be counted as only one event
                isolated_false = np.where(~bloom_condition)[0]
                for idx in isolated_false:
                    if idx > 0 and idx < len(bloom_condition) - 1:
                        if bloom_condition[idx - 1] and bloom_condition[idx + 1]:
                            bloom_condition[idx] = True
                        elif idx > 1 and idx < len(bloom_condition) - 2:
                            # Check if there are two or fewer isolated days (2 False values between True values)
                            if bloom_condition[idx - 2] and bloom_condition[idx + 2]:
                                bloom_condition[idx] = True
                
                #Count of blooms
                blooms_labeled, num_blooms = label(bloom_condition)
                
                # Find indices where bloom conditions are True
                bloom_indices = np.where(bloom_condition)[0]
                
                if bloom_indices > 0:
                    
                    # Update metrics if blooms are detected
                    bloom_freq[i, j, year] = num_blooms
                    bloom_max[i, j, year] = np.max(pixel_series[bloom_indices])
                    bloom_peak[i, j, year] = bloom_indices[np.argmax(pixel_series[bloom_indices])]
                    bloom_amp[i, j, year] = bloom_max[i, j, year] - mean_value
                    bloom_ini[i, j, year], bloom_fin[i, j, year] = bloom_detect(bloom_indices, bloom_peak[i, j, year]) 
                    bloom_dur[i, j, year] = bloom_fin[i, j, year] - bloom_ini[i, j, year]
                    bloom_main_cum[i, j, year] = np.trapz(pixel_series[int(bloom_ini[i, j, year]):int(bloom_fin[i, j, year]) + 1])
                    bloom_total_cum[i, j, year] = np.trapz(pixel_series[bloom_indices])
                    


                    
# Save metrics so far (in .npy)
directory = r'E:\...\Phenology_Metrics\CAN/'

np.save(directory+'LAT_CAN.npy', LAT)
np.save(directory+'LON_CAN.npy', LON)

np.save(directory+'bloom_freq_CAN.npy', bloom_freq)
np.save(directory+'bloom_max_CAN.npy', bloom_max)
np.save(directory+'bloom_peak_CAN.npy', bloom_peak)
np.save(directory+'bloom_amp_CAN.npy', bloom_amp)
np.save(directory+'bloom_ini_CAN.npy', bloom_ini)
np.save(directory+'bloom_fin_CAN.npy', bloom_fin)
np.save(directory+'bloom_dur_CAN.npy', bloom_dur)
np.save(directory+'bloom_main_cum_CAN.npy', bloom_main_cum)
np.save(directory+'bloom_total_cum_CAN.npy', bloom_total_cum)                           


###############################################################################
# Load previously-calculated datasets for mean metrics
directory = r'E:\...\Phenology_Metrics\CAN/'

LAT_CAN = np.load(directory + 'LAT_NA.npy')
LON_CAN = np.load(directory + 'LON_NA.npy')

bloom_freq = np.load(directory + 'bloom_freq_CAN.npy')
bloom_max = np.load(directory + 'bloom_max_CAN.npy')
bloom_peak = np.load(directory + 'bloom_peak_CAN.npy')
bloom_amp = np.load(directory + 'bloom_amp_CAN.npy')
bloom_ini = np.load(directory + 'bloom_ini_CAN.npy')
bloom_fin = np.load(directory + 'bloom_fin_CAN.npy')
bloom_dur = np.load(directory + 'bloom_dur_CAN.npy')
bloom_main_cum = np.load(directory + 'bloom_main_cum_CAN.npy')
bloom_total_cum = np.load(directory + 'bloom_total_cum_CAN.npy')




                    ##General average##

metrics = [bloom_freq, bloom_max, bloom_peak, bloom_amp, bloom_ini, bloom_fin, bloom_dur, bloom_main_cum]

mean_metrics = [np.nanmean(metric[:,:,:], axis=2) for metric in metrics]

fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
fig.suptitle('1998-2023', fontsize=20)                       

metric_names = ['Bloom Frequency', 'Bloom Max', 'Bloom Peak', 'Bloom Amplitude', 'Bloom Start', 'Bloom End', 'Bloom Duration', 'Bloom Cumulative Chl-a']

for mean_metric, name, ax in zip(mean_metrics, metric_names, axes.flat):

    im = ax.pcolormesh(LON_CAN, LAT_CAN, mean_metric, cmap='Spectral_r', shading='auto')
    ax.set_title(name)
    
    cbar = fig.colorbar(im, ax=ax, pad=0.02, label='')

plt.tight_layout()
plt.show()
# Save figure
fig.savefig(r'E:\.../MeanMetrics_CAN.png', dpi=600)



###############################################################################

## Calculating Trends ##

# Firstly, load previously-calculated datasets for mean metrics
directory = r'E:\...\Phenology_Metrics\BAL/'

bloom_freq = np.load(directory + 'bloom_freq_BAL.npy')
bloom_max = np.load(directory + 'bloom_max_BAL.npy')
bloom_peak = np.load(directory + 'bloom_peak_BAL.npy')
bloom_amp = np.load(directory + 'bloom_amp_BAL.npy')
bloom_ini = np.load(directory + 'bloom_ini_BAL.npy')
bloom_fin = np.load(directory + 'bloom_fin_BAL.npy')
bloom_dur = np.load(directory + 'bloom_dur_BAL.npy')
bloom_main_cum = np.load(directory + 'bloom_main_cum_BAL.npy')
bloom_total_cum = np.load(directory + 'bloom_total_cum_BAL.npy')



def CHL_trend(array):
    # Obtaining array dimensions
    number_years = array.shape[2]
    
    # Create arrays to store decadal trends and significance
    decadal_trends = np.zeros((array.shape[0], array.shape[1]))
    significance = np.zeros((array.shape[0], array.shape[1]))
    
    # Computing the decadal trend and significance for each pixel
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # Retrieve the time series values for the current pixel
            temporal_serie = array[i, j, :]
            
            # Calculate the years corresponding to each data point
            years = np.arange(0, number_years)
            
            # Finding indeces for NaN values
            indices_no_nan = np.isfinite(temporal_serie)
            
            if np.sum(indices_no_nan) > 1:
                # If there are at least 2 non-NaN points, calculate the linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(years[indices_no_nan], temporal_serie[indices_no_nan])
                
                # Computing decadal trend
                decadal_trend = slope * 10
                
                # Computing significance
                if p_value >= 0.05:
                    significance_value = np.nan
                else:
                    significance_value = 1
            else:
                # If there are not enough non-NaN points, set the decadal trend and significance as NaN
                decadal_trend = np.nan
                significance_value = np.nan
            
            # Store the decadal trend and significance in the result arrays
            decadal_trends[i, j] = decadal_trend
            significance[i, j] = significance_value
    
    return decadal_trends, significance



# Calculating decadal trends and significance
bloom_freq_trend_BAL, bloom_freq_significance_BAL = CHL_trend(bloom_freq)

bloom_max_trend_BAL, bloom_max_significance_BAL = CHL_trend(bloom_max)

bloom_peak_trend_BAL, bloom_peak_significance_BAL = CHL_trend(bloom_peak)

bloom_amp_trend_BAL, bloom_amp_significance_BAL = CHL_trend(bloom_amp)

bloom_ini_trend_BAL, bloom_ini_significance_BAL = CHL_trend(bloom_ini)

bloom_fin_trend_BAL, bloom_fin_significance_BAL = CHL_trend(bloom_fin)

bloom_dur_trend_BAL, bloom_dur_significance_BAL = CHL_trend(bloom_dur)

bloom_main_cum_trend_BAL, bloom_main_cum_significance_BAL = CHL_trend(bloom_main_cum)

bloom_total_cum_trend_BAL, bloom_total_cum_significance_BAL = CHL_trend(bloom_total_cum)


#Save trends and significance so far
directory = r'E:\...\Phenology_Metrics\BAL/'

np.save(directory+'bloom_freq_trend_BAL.npy', bloom_freq_trend_BAL)
np.save(directory+'bloom_freq_significance_BAL.npy', bloom_freq_significance_BAL)

np.save(directory+'bloom_max_trend_BAL.npy', bloom_max_trend_BAL)
np.save(directory+'bloom_max_significance_BAL.npy', bloom_max_significance_BAL)

np.save(directory+'bloom_peak_trend_BAL.npy', bloom_peak_trend_BAL)
np.save(directory+'bloom_peak_significance_BAL.npy', bloom_peak_significance_BAL)

np.save(directory+'bloom_amp_trend_BAL.npy', bloom_amp_trend_BAL)
np.save(directory+'bloom_amp_significance_BAL.npy', bloom_amp_significance_BAL)

np.save(directory+'bloom_ini_trend_BAL.npy', bloom_ini_trend_BAL)
np.save(directory+'bloom_ini_significance_BAL.npy', bloom_ini_significance_BAL)

np.save(directory+'bloom_fin_trend_BAL.npy', bloom_fin_trend_BAL)
np.save(directory+'bloom_fin_significance_BAL.npy', bloom_fin_significance_BAL)

np.save(directory+'bloom_dur_trend_BAL.npy', bloom_dur_trend_BAL)
np.save(directory+'bloom_dur_significance_BAL.npy', bloom_dur_significance_BAL)

np.save(directory+'bloom_main_cum_trend_BAL.npy', bloom_main_cum_trend_BAL)
np.save(directory+'bloom_main_cum_significance_BAL.npy', bloom_main_cum_significance_BAL)

np.save(directory+'bloom_total_cum_trend_BAL.npy', bloom_total_cum_trend_BAL)
np.save(directory+'bloom_total_cum_significance_BAL.npy', bloom_total_cum_significance_BAL)

###############################################################################





# Load previously-calculated datasets for mean metrics
directory = r'E:\...\Phenology_Metrics\GC/'

bloom_freq_trend_GC = np.load(directory + 'bloom_freq_trend_GC.npy')
bloom_freq_significance_GC = np.load(directory + 'bloom_freq_significance_GC.npy')

bloom_max_trend_GC = np.load(directory + 'bloom_max_trend_GC.npy')
bloom_max_significance_GC = np.load(directory + 'bloom_max_significance_GC.npy')

bloom_peak_trend_GC = np.load(directory + 'bloom_peak_trend_GC.npy')
bloom_peak_significance_GC = np.load(directory + 'bloom_peak_significance_GC.npy')

bloom_amp_trend_GC = np.load(directory + 'bloom_amp_trend_GC.npy')
bloom_amp_significance_GC = np.load(directory + 'bloom_amp_significance_GC.npy')

bloom_ini_trend_GC = np.load(directory + 'bloom_ini_trend_GC.npy')
bloom_ini_significance_GC = np.load(directory + 'bloom_ini_significance_GC.npy')

bloom_fin_trend_GC = np.load(directory + 'bloom_fin_trend_GC.npy')
bloom_fin_significance_GC = np.load(directory + 'bloom_fin_significance_GC.npy')

bloom_dur_trend_GC = np.load(directory + 'bloom_dur_trend_GC.npy')
bloom_dur_significance_GC = np.load(directory + 'bloom_dur_significance_GC.npy')

bloom_main_cum_trend_GC = np.load(directory + 'bloom_main_cum_trend_GC.npy')
bloom_main_cum_significance_GC = np.load(directory + 'bloom_main_cum_significance_GC.npy')

bloom_total_cum_trend_GC = np.load(directory + 'bloom_total_cum_trend_GC.npy')
bloom_total_cum_significance_GC = np.load(directory + 'bloom_total_cum_significance_GC.npy')



############################## ##TRENDS## #####################################

metric_values = [bloom_freq_trend_GC, bloom_max_trend_GC, bloom_peak_trend_GC, 
                  bloom_amp_trend_GC, bloom_ini_trend_GC, bloom_fin_trend_GC, 
                  bloom_dur_trend_GC, bloom_main_cum_trend_GC]

signif_metrics = [bloom_freq_significance_GC, bloom_max_significance_GC, bloom_peak_significance_GC, 
                  bloom_amp_significance_GC, bloom_ini_significance_GC, bloom_fin_significance_GC, 
                  bloom_dur_significance_GC, bloom_main_cum_significance_GC]

metric_names = ['Bloom Frequency Trend', 'Bloom Max Trend', 'Bloom Peak Trend', 'Bloom Amplitude Trend', 'Bloom Start Trend', 'Bloom End Trend', 'Bloom Duration Trend', 'Bloom Cumulative Chl-a Trend']

fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
fig.suptitle('Decadal Trends [1998-2023]', fontsize=20)

for metrics, signif, name, ax in zip(metric_values, signif_metrics, metric_names, axes.flat):
   
    norm = TwoSlopeNorm(vmin=np.nanmin(metrics), vcenter=0, vmax=np.nanmax(metrics))

    im = ax.pcolormesh(LON_GC, LAT_GC, metrics, cmap='PRGn', norm=norm, shading='auto')
    ax.set_title(name)
    
    ax.scatter(LON_GC[::6, ::6], LAT_GC[::6, ::6], signif[::6, ::6], color='k', marker='o', alpha=0.8, linewidth=2)
    
    cbar = fig.colorbar(im, ax=ax, pad=0.02, label='')

plt.tight_layout()
plt.show()

# Save figure
fig.savefig(r'E:\...\Fig_Test/Trends_GC.png', dpi=600)

###############################################################################

