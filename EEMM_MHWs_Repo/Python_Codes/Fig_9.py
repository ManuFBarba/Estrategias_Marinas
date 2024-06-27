# -*- coding: utf-8 -*-
"""

#########################      Figure 9 in 
Fernández-Barba, M., Huertas, I. E., & Navarro, G. (2024). 
Assessment of surface and bottom marine heatwaves along the Spanish coast. 
Ocean Modelling, 190, 102399.                          ########################

"""

#Loading required libraries
import numpy as np
# import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import cartopy.crs as ccrs
import cartopy.feature as cft
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

##Load MHWs_from_MATLAB.py
##Load Datos_ESMARES.py
##Load 'synchrony_matrices' from Fig_8.py


                #################################
                ## Integrated Exposure to MHWs ##
                #################################

##Cumulative Intensity Exposure (measured based on whether the average cumulative intensity value for the entire demarcation is exceeded in the oceanic pixel, both at the surface and at the bottom, throughout the study period.)
# Define a dictionary to store MHW type data
CumInt_MHW = {
    "NA": {"BMHW": BMHW_cum_NA_MODEL, "MHW": MHW_cum_NA_MODEL},
    "AL": {"BMHW": BMHW_cum_AL_MODEL, "MHW": MHW_cum_AL_MODEL},
    "CAN": {"BMHW": BMHW_cum_CAN_MODEL, "MHW": MHW_cum_CAN_MODEL},
    "GC": {"BMHW": BMHW_cum_SA_MODEL, "MHW": MHW_cum_SA_MODEL},
    "BAL": {"BMHW": BMHW_cum_BAL_MODEL, "MHW": MHW_cum_BAL_MODEL}
}

# Initialize results dictionaries
CumInt_Exposure = {}
Integrated_Exposure_to_MHWs = {}

# Function to calculate mean values and exposures
def calculate_CumInt_exposure(CumInt_data):
    bottom_mean = np.round(np.nanmean(CumInt_data["BMHW"]))
    surface_mean = np.round(np.nanmean(CumInt_data["MHW"]))
    
    CumInt_exposure = np.full_like(CumInt_data["MHW"], np.nan)
    CumInt_exposure[(CumInt_data["BMHW"] > bottom_mean) & (CumInt_data["MHW"] > surface_mean)] = 1.0
    CumInt_exposure[(CumInt_data["BMHW"] < bottom_mean) & (CumInt_data["MHW"] >= surface_mean)] = 0.75
    CumInt_exposure[(CumInt_data["BMHW"] >= bottom_mean) & (CumInt_data["MHW"] < surface_mean)] = 0.75
    CumInt_exposure[(np.round(CumInt_data["BMHW"]) == bottom_mean) & (np.round(CumInt_data["MHW"]) == surface_mean)] = 0.5
    CumInt_exposure[(CumInt_data["BMHW"] < bottom_mean) & (CumInt_data["MHW"] < surface_mean)] = 0.25
    
    return CumInt_exposure

# Calculate CumInt_Exposure for each region
for region, CumInt_data in CumInt_MHW.items():
    CumInt_Exposure[region] = calculate_CumInt_exposure(CumInt_data)


##Integrated Exposure to MHWs along SMDs (Overlap of tercile categories)
# Function to calculate Integrated Exposure to MHWs based on Cumulative Intensity Exposure along with Synchrony matrices
def calculate_integrated_exposure(region, CumInt_exposure, Synchrony):
    Integrated_exposure = np.full_like(CumInt_exposure, np.nan)
    Integrated_exposure[(CumInt_exposure == 1.00) & (Synchrony >= 0.60)] = 1.00
    Integrated_exposure[(CumInt_exposure == 1.00) & (Synchrony < 0.60) & (Synchrony >= 0.30)] = 0.90
    Integrated_exposure[(CumInt_exposure == 1.00) & (Synchrony < 0.30)] = 0.80
    Integrated_exposure[(CumInt_exposure <= 0.75) & (CumInt_exposure > 0.25) & (Synchrony < 0.30)] = 0.70
    Integrated_exposure[(CumInt_exposure <= 0.75) & (CumInt_exposure > 0.25) & (Synchrony < 0.60) & (Synchrony >= 0.30)] = 0.60
    Integrated_exposure[(CumInt_exposure <= 0.75) & (CumInt_exposure > 0.25) & (Synchrony >= 0.60)] = 0.50
    Integrated_exposure[(CumInt_exposure == 0.25) & (Synchrony >= 0.60)] = 0.40
    Integrated_exposure[(CumInt_exposure == 0.25) & (Synchrony < 0.60) & (Synchrony >= 0.30)] = 0.30
    Integrated_exposure[(CumInt_exposure == 0.25) & (Synchrony < 0.30)] = 0.20
    
    return Integrated_exposure

# Calculate Total_Exposure for each region
for region in CumInt_MHW.keys():
    Integrated_Exposure_to_MHWs[region] = calculate_integrated_exposure(region, CumInt_Exposure[region], synchrony_matrices[region])


###############################################################################
# ##Save Integrated Exposure to MHWs datasets so far
# #Extracting synchrony matrices from the dictionary
# Integrated_Exposure_NA = Integrated_Exposure_to_MHWs['NA']
# Integrated_Exposure_AL = Integrated_Exposure_to_MHWs['AL']
# Integrated_Exposure_CAN = Integrated_Exposure_to_MHWs['CAN']
# Integrated_Exposure_SA = Integrated_Exposure_to_MHWs['SA']
# Integrated_Exposure_BAL = Integrated_Exposure_to_MHWs['BAL']

# #Extracting lon and lat arrays for each SMD
# lon_NA = lon_NA_MODEL[:, 0]
# lat_NA = lat_NA_MODEL[:, 0]
# lon_AL = lon_AL_MODEL[:, 0]
# lat_AL = lat_AL_MODEL[:, 0]
# lon_CAN = lon_CAN_MODEL[:, 0]
# lat_CAN = lat_CAN_MODEL[:, 0]
# lon_SA = lon_SA_MODEL[:, 0]
# lat_SA = lat_SA_MODEL[:, 0]
# lon_BAL = lon_BAL_MODEL[:, 0]
# lat_BAL = lat_BAL_MODEL[:, 0]


# # Create data_vars (xarray) for each variable using their lat and lon as coords
# data_vars = {
#     'Integrated_Exposure_NA': xr.DataArray(Integrated_Exposure_NA, dims=['lon_NA', 'lat_NA'], coords={'lon_NA': lon_NA, 'lat_NA': lat_NA}),
#     'Integrated_Exposure_AL': xr.DataArray(Integrated_Exposure_AL, dims=['lon_AL', 'lat_AL'], coords={'lon_AL': lon_AL, 'lat_AL': lat_AL}),
#     'Integrated_Exposure_CAN': xr.DataArray(Integrated_Exposure_CAN, dims=['lon_CAN', 'lat_CAN'], coords={'lon_CAN': lon_CAN, 'lat_CAN': lat_CAN}),
#     'Integrated_Exposure_SA': xr.DataArray(Integrated_Exposure_SA, dims=['lon_SA', 'lat_SA'], coords={'lon_SA': lon_SA, 'lat_SA': lat_SA}),
#     'Integrated_Exposure_BAL': xr.DataArray(Integrated_Exposure_BAL, dims=['lon_BAL', 'lat_BAL'], coords={'lon_BAL': lon_BAL, 'lat_BAL': lat_BAL}),
#     }


# ## Creating Ds
# ds = xr.Dataset(data_vars)

# ds = ds.transpose('lat_NA', 'lon_NA', 'lat_AL', 'lon_AL', 'lat_CAN', 'lon_CAN', 'lat_SA', 'lon_SA', 'lat_BAL', 'lon_BAL')

# # Adding metadata to Ds
# ds.attrs['description'] = 'Integrated Exposure to MHWs (1993-2022) along each Spanish Marine Demarcation'
# ds.attrs['source'] = 'Fernández-Barba et al. (2024)'

# # Adding metadata to each variable
# ds['Integrated_Exposure_NA'].attrs['units'] = ''
# ds['Integrated_Exposure_NA'].attrs['long_name'] = 'Integrated Exposure to MHWs in North Atlantic (NA) Demarcation'

# ds['Integrated_Exposure_AL'].attrs['units'] = ''
# ds['Integrated_Exposure_AL'].attrs['long_name'] = 'Integrated Exposure to MHWs in SoG and Alboran Sea (AL) Demarcation'

# ds['Integrated_Exposure_CAN'].attrs['units'] = ''
# ds['Integrated_Exposure_CAN'].attrs['long_name'] = 'Integrated Exposure to MHWs in Canary (CAN) Demarcation'

# ds['Integrated_Exposure_SA'].attrs['units'] = ''
# ds['Integrated_Exposure_SA'].attrs['long_name'] = 'Integrated Exposure to MHWs in South Atlantic (SA) Demarcation'

# ds['Integrated_Exposure_BAL'].attrs['units'] = ''
# ds['Integrated_Exposure_BAL'].attrs['long_name'] = 'Integrated Exposure to MHWs in Levantine-Balearic (BAL) Demarcation'

# ds.to_netcdf(r'...\Syn_Exposure_Datasets/Integrated_Exposure_to_MHWs.nc')
###############################################################################


## Integrated Exposure to MHWs Mapplot (Fig. 9)         
fig = plt.figure(figsize=(20, 10))

proj = ccrs.PlateCarree()  # Choose the projection
# Adding some cartopy natural features
land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
                                   edgecolor='none', facecolor='black', linewidth=0.5)
axs = plt.axes(projection=proj)
# Set tick font size
for label in (axs.get_xticklabels() + axs.get_yticklabels()):
    label.set_fontsize(20)

cmap=plt.cm.Spectral_r
levels=np.linspace(0.2, 1, num=10)

cs_1= axs.contourf(LON_SA_MODEL, LAT_SA_MODEL, Integrated_Exposure_to_MHWs['GC'], levels, cmap=cmap, transform=proj, extend ='neither')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, Integrated_Exposure_to_MHWs['AL'], levels, cmap=cmap, transform=proj, extend ='neither')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, Integrated_Exposure_to_MHWs['BAL'], levels, cmap=cmap, transform=proj, extend ='neither')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, Integrated_Exposure_to_MHWs['NA'], levels, cmap=cmap, transform=proj, extend ='neither')

# Define contour levels for elevation and plot elevation isolines
elev_levels = [400, 2500]

el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_1 in el_1.collections:
    line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_2 in el_2.collections:
    line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_3 in el_3.collections:
    line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
                  linestyles=['solid', 'dashed'], linewidths=1.25)
for line_4 in el_4.collections:
    line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


cbar = plt.colorbar(cs_1, ax=axs, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(axis='y', size=10, direction='out', labelsize=22)
cbar.ax.minorticks_off()
cbar.set_ticks([0.335, 0.515, 0.695, 0.87])
cbar.set_ticklabels(['L. Cum Int & L. Syn', 'L. Cum Int & H. Syn', 'H. Cum Int & L. Syn', 'H. Cum Int & H. Syn'])


# Add map features
axs.coastlines(resolution='10m', color='black', linewidth=1)
axs.add_feature(land_10m)

# Set the extent of the main plot
axs.set_extent([-16, 6.5, 34, 47], crs=proj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
axs.xaxis.set_major_formatter(lon_formatter)
axs.yaxis.set_major_formatter(lat_formatter)
axs.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
axs.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)
#Set the title
axs.set_title(r'Integrated exposure to MHWs', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, Integrated_Exposure_to_MHWs['CAN'], levels, cmap=cmap, transform=proj, extend ='neither')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'...\Fig_9\Integrated_Exposure_to_MHWs.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




## Ocean cells percent normalized to 100%

def calculate_percentages(data, variable):
    # Applying the conditions to obtain values within the specified intervals
    LL_cond = (data[variable] >= 0.2) & (data[variable] < 0.4)
    LH_cond = (data[variable] >= 0.4) & (data[variable] < 0.6)
    HL_cond = (data[variable] >= 0.6) & (data[variable] < 0.8)
    HH_cond = (data[variable] >= 0.8) & (data[variable] <= 1)

    # Calculating percentage for each Category
    LL = (np.sum(LL_cond) / np.sum(~np.isnan(data[variable]))) * 100
    LH = (np.sum(LH_cond) / np.sum(~np.isnan(data[variable]))) * 100
    HL = (np.sum(HL_cond) / np.sum(~np.isnan(data[variable]))) * 100
    HH = (np.sum(HH_cond) / np.sum(~np.isnan(data[variable]))) * 100

    return LL, LH, HL, HH

variables = ['BAL', 'GC', 'CAN', 'AL', 'NA']

results = {}

for var in variables:
    LL_var, LH_var, HL_var, HH_var = calculate_percentages(Integrated_Exposure_to_MHWs, var)
    results[var] = {
        'LL': LL_var,
        'LH': LH_var,
        'HL': HL_var,
        'HH': HH_var
    }

    print(f"% LL_{var}:", LL_var)
    print(f"% LH_{var}:", LH_var)
    print(f"% HL_{var}:", HL_var)
    print(f"% HH_{var}:", HH_var)

LL_values = [results[var]['LL'] for var in variables]
LH_values = [results[var]['LH'] for var in variables]
HL_values = [results[var]['HL'] for var in variables]
HH_values = [results[var]['HH'] for var in variables]

regions = variables

colors = ['#466EB1', '#9CD7A4', '#FBA05B', '#BC2249']

bar_width = 0.75

fig, ax3 = plt.subplots(figsize=(15, 5))

bars1 = ax3.barh(regions, LL_values, color=colors[0])
bars2 = ax3.barh(regions, LH_values, left=LL_values, color=colors[1])
bars3 = ax3.barh(regions, HL_values, left=np.add(LL_values, LH_values), color=colors[2])
bars4 = ax3.barh(regions, HH_values, left=np.add(LL_values, np.add(LH_values, HL_values)), color=colors[3])

ax3.tick_params(axis='both', which='major', length=10, labelsize=15)
ax3.set_xlabel('Ocean cells (%)', fontsize=16)
ax3.set_ylabel('SMD', fontsize=16)
ax3.set_xlim(0, 100)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

outfile = r'...\Fig_9\Ocean_Cells_Percent.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')
