# -*- coding: utf-8 -*-
"""

################################ SCR metric  ##################################

"""

#Loading required libraries
import numpy as np
import xarray as xr 

from scipy import stats

from tqdm import tqdm


# Load dataset
ds = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_BAL.nc')

# Calculate daily climatology
climatology = ds.groupby('time.dayofyear').mean(dim='time')

# Define intervals for each year
last_year = "2023"
year_intervals = [(f"{year}-01-01", f"{year}-12-31") for year in range(1998, 2024)]
# year_intervals[-1] = (f"{last_year}-01-01", f"{last_year}-11-10")  # Adjust last interval when last year is incomplete

# Matrix sizes
lat_size = ds.sizes['lat']
lon_size = ds.sizes['lon']
year_intervals_size = len(year_intervals)

# Initialize SCR array with np.NaNs
SCR = np.full((lat_size, lon_size, year_intervals_size), np.NaN)

# Loop over the years
for year, (start_date, end_date) in enumerate(tqdm(year_intervals, desc='Processing years')):
    # Filter data for the current year
    ds_year = ds.sel(time=slice(start_date, end_date))

    # Verify if ds_year has any data
    if ds_year.time.size == 0:
        continue  # Skip to the next year if there is no data for the current year

    # Loop over the grid cells
    for i in tqdm(range(lat_size), desc='Processing rows', leave=False):
        for j in tqdm(range(lon_size), desc='Processing columns', leave=False):
            # Get the time series for the pixel
            pixel_series = ds_year.isel(lat=i, lon=j).CHL.values.squeeze()
            seasonal_mean = climatology.isel(lat=i, lon=j).CHL.values.squeeze()

            # Verify if all values in the series are NaN
            if not np.all(np.isnan(pixel_series)):
                # Ensure both series have the same length (excluding day 366 for years that do not have it)
                min_length = min(len(pixel_series), len(seasonal_mean))
                pixel_series = pixel_series[:min_length]
                seasonal_mean = seasonal_mean[:min_length]

                # Calculate Pearson correlation coefficient using np.ma.corrcoef (ignoring np.NaN values)
                corr_matrix = np.ma.corrcoef(np.ma.masked_invalid(pixel_series), np.ma.masked_invalid(seasonal_mean))

                # Extract the correlation coefficient from the matrix
                SCR[i, j, year] = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else np.NaN


# Save metrics so far (in .npy)
directory = r'E:\...\Phenology_Metrics\BAL/'
np.save(directory+'SCR_BAL.npy', SCR)


# plt.plot(np.arange(1998, 2024), np.nanmean(SCR, axis=(0,1)))


# ########################################################
# ds_year = ds.sel(time=slice("2012-01-01", "2012-12-31"))
# pixel_series = ds_year.sel(lat=36.5, lon=-6.5, method='nearest').CHL.values.squeeze()
# seasonal_mean = climatology.sel(lat=36.5, lon=-6.5, method='nearest').CHL.values.squeeze()

# plt.plot(pixel_series, color='g')
# plt.plot(seasonal_mean, color='k')
# plt.legend(['2012', 'climatology'])
# #######################################################











## Calculating Trends ##

# Firstly, load previously-calculated datasets for mean metrics
directory = r'E:\...\Phenology_Metrics\BAL/'

SCR_BAL = np.load(directory + 'SCR_BAL.npy')


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

SCR_trend_BAL, SCR_significance_BAL = CHL_trend(SCR_BAL)


#Save trends and significance so far
directory = r'E:\...\Phenology_Metrics\BAL/'

np.save(directory+'SCR_trend_BAL.npy', SCR_trend_BAL)
np.save(directory+'SCR_significance_BAL.npy', SCR_significance_BAL)

###############################################################################                  