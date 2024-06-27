# -*- coding: utf-8 -*-
"""

#########################      Figure 8 in 
Fernández-Barba, M., Huertas, I. E., & Navarro, G. (2024). 
Assessment of surface and bottom marine heatwaves along the Spanish coast. 
Ocean Modelling, 190, 102399.                          ########################

"""

#Loading required libraries
import numpy as np
import xarray as xr 

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import cmocean as cm

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import cartopy.feature as cft

from scipy.stats import spearmanr


##Load MHWs_from_MATLAB.py
##Load Datos_ESMARES.py



                ##############################
                ##  SYNCHRONY BMHWs - SMHWs ##
                ##############################


# Find the indices where there are no NaNs in both matrices
valid_indices_SA = ~np.isnan(MHW_td_SA_MODEL)
valid_indices_AL = ~np.isnan(MHW_td_AL_MODEL)
valid_indices_BAL = ~np.isnan(MHW_td_BAL_MODEL)
valid_indices_NA = ~np.isnan(MHW_td_NA_MODEL)
valid_indices_CAN = ~np.isnan(MHW_td_CAN_MODEL)

# Create a dictionary to store the matrices
synchrony_matrices = {}

# List of regions
regions = ["GC", "AL", "BAL", "NA", "CAN"]

# Loop through each region
for region in regions:
    BMHW_td_ts = locals()["BMHW_td_ts_" + region + "_MODEL_monthly"]
    MHW_td_ts = locals()["MHW_td_ts_" + region + "_MODEL_monthly"]
    valid_indices = locals()["valid_indices_" + region]
    
    Synchrony_Matrix = np.zeros(BMHW_td_ts.shape[:2])
    
    # Calculate the Spearman correlation
    for lon in range(BMHW_td_ts.shape[0]):
        for lat in range(BMHW_td_ts.shape[1]):
            if np.all(valid_indices[lon, lat]):  # Check if all elements in the subarray are True
                if np.any(np.isnan(BMHW_td_ts[lon, lat, :])) or np.any(np.isnan(MHW_td_ts[lon, lat, :])):
                    Synchrony_Matrix[lon, lat] = 0  # Set synchrony to 0 if there are any NaNs
                else:
                    corr, _ = spearmanr(
                        BMHW_td_ts[lon, lat, :],
                        MHW_td_ts[lon, lat, :]
                    )
                    Synchrony_Matrix[lon, lat] = corr
    
    # Replace < 0 with 0 in valid_indices = True
    Synchrony_Matrix[(Synchrony_Matrix < 0) & valid_indices] = 0
    
    # Set values outside valid_indices to np.NaN
    Synchrony_Matrix[~valid_indices] = np.nan
    
    # Update the Synchrony_Matrix in synchrony_matrices
    synchrony_matrices[region] = Synchrony_Matrix


###############################################################################
# ##Save synchrony datasets so far
# #Extracting synchrony matrices from the dictionary
# Synchrony_NA = synchrony_matrices['NA']
# Synchrony_AL = synchrony_matrices['AL']
# Synchrony_CAN = synchrony_matrices['CAN']
# Synchrony_SA = synchrony_matrices['GC']
# Synchrony_BAL = synchrony_matrices['BAL']

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
#     'BMHW_SMHW_Synchrony_NA': xr.DataArray(Synchrony_NA, dims=['lon_NA', 'lat_NA'], coords={'lon_NA': lon_NA, 'lat_NA': lat_NA}),
#     'BMHW_SMHW_Synchrony_AL': xr.DataArray(Synchrony_AL, dims=['lon_AL', 'lat_AL'], coords={'lon_AL': lon_AL, 'lat_AL': lat_AL}),
#     'BMHW_SMHW_Synchrony_CAN': xr.DataArray(Synchrony_CAN, dims=['lon_CAN', 'lat_CAN'], coords={'lon_CAN': lon_CAN, 'lat_CAN': lat_CAN}),
#     'BMHW_SMHW_Synchrony_SA': xr.DataArray(Synchrony_SA, dims=['lon_SA', 'lat_SA'], coords={'lon_SA': lon_SA, 'lat_SA': lat_SA}),
#     'BMHW_SMHW_Synchrony_BAL': xr.DataArray(Synchrony_BAL, dims=['lon_BAL', 'lat_BAL'], coords={'lon_BAL': lon_BAL, 'lat_BAL': lat_BAL}),
#     }


# ## Creating Ds
# ds = xr.Dataset(data_vars)

# ds = ds.transpose('lat_NA', 'lon_NA', 'lat_AL', 'lon_AL', 'lat_CAN', 'lon_CAN', 'lat_SA', 'lon_SA', 'lat_BAL', 'lon_BAL')

# # Adding metadata to Ds
# ds.attrs['description'] = 'Surface (SMHW) and Bottom (BMHW) Synchrony along each Spanish Marine Demarcation'
# ds.attrs['source'] = 'Fernández-Barba et al. (2024)'

# # Adding metadata to each variable
# ds['BMHW_SMHW_Synchrony_NA'].attrs['units'] = 'ρ'
# ds['BMHW_SMHW_Synchrony_NA'].attrs['long_name'] = 'BMHW & SMHW Synchrony in North Atlantic (NA) Demarcation'

# ds['BMHW_SMHW_Synchrony_AL'].attrs['units'] = 'ρ'
# ds['BMHW_SMHW_Synchrony_AL'].attrs['long_name'] = 'BMHW & SMHW Synchrony in SoG and Alboran Sea (AL) Demarcation'

# ds['BMHW_SMHW_Synchrony_CAN'].attrs['units'] = 'ρ'
# ds['BMHW_SMHW_Synchrony_CAN'].attrs['long_name'] = 'BMHW & SMHW Synchrony in Canary (CAN) Demarcation'

# ds['BMHW_SMHW_Synchrony_SA'].attrs['units'] = 'ρ'
# ds['BMHW_SMHW_Synchrony_SA'].attrs['long_name'] = 'BMHW & SMHW Synchrony in South Atlantic (SA) Demarcation'

# ds['BMHW_SMHW_Synchrony_BAL'].attrs['units'] = 'ρ'
# ds['BMHW_SMHW_Synchrony_BAL'].attrs['long_name'] = 'BMHW & SMHW Synchrony in Levantine-Balearic (BAL) Demarcation'

# ds.to_netcdf(r'...\Datasets/BMHW_SMHW_Synchrony.nc')
# ###############################################################################


## BMHW & SMHW Synchrony Mapplot (Figure 8a)         
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

levels=np.linspace(0, 1, num=11)

cs_1= axs.contourf(LON_SA_MODEL, LAT_SA_MODEL, synchrony_matrices['GC'], levels, cmap=cmap, transform=proj, extend ='neither')
cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, synchrony_matrices['AL'], levels, cmap=cmap, transform=proj, extend ='neither')
cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, synchrony_matrices['BAL'], levels, cmap=cmap, transform=proj, extend ='neither')
cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, synchrony_matrices['NA'], levels, cmap=cmap, transform=proj, extend ='neither')

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
cbar.ax.tick_params(axis='y', size=10, direction='out', labelsize=18)
cbar.ax.minorticks_off()
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_label(r'BMHW & SMHW Synchrony', fontsize=18)
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
# axs.set_title(r'BMHW & SMHW Synchrony', fontsize=25)

# Create a second set of axes (subplots) for the Canarian box
box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)
# Modify the [left, bottom, width, height] values in the line above to adjust the position and size.

# Plot the data for Canarias on the second axes
cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, synchrony_matrices['CAN'], levels, cmap=cmap, transform=proj, extend ='neither')
el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
                      linestyles=['solid', 'dashed'], linewidths=1.25)
for line_5 in el_5.collections:
    line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# Add map features for the second axes
box_ax.coastlines(resolution='10m', color='black', linewidth=1)
box_ax.add_feature(land_10m)


outfile = r'...\Fig_8\8a_BMHW_SMHW_Synchrony.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')

###############################################################################




## 2D Probability BMHW & SMHW Synchrony Histograms (Figures 8b-f)


                    ##############################
                    ##  Max MLT / Bottom Depth  ##
                    ##############################

#Load Max MLT data#
Max_MLT_SA = np.load(r'...\Max_MLT_SA.npy')
Max_MLT_AL = np.load(r'...\Max_MLT_AL.npy')
Max_MLT_BAL = np.load(r'...\Max_MLT_BAL.npy')
Max_MLT_NA = np.load(r'...\Max_MLT_NA.npy')
Max_MLT_CAN = np.load(r'...\Max_MLT_CAN.npy')


#Interpolate Bathymetry grid to the GLORYS dataset grid
#Elevation_Interp
new_lat_SA = np.asarray(ds_Model_SA.latitude)
new_lon_SA = np.asarray(ds_Model_SA.longitude)
elevation_SA_interp = elevation_SA.interp(lat=new_lat_SA, lon=new_lon_SA)
elevation_SA_interp = elevation_SA_interp.rename({'lat': 'latitude'})
elevation_SA_interp = elevation_SA_interp.rename({'lon': 'longitude'})

new_lat_AL = np.asarray(ds_Model_AL.latitude)
new_lon_AL = np.asarray(ds_Model_AL.longitude)
elevation_AL_interp = elevation_AL.interp(lat=new_lat_AL, lon=new_lon_AL)
elevation_AL_interp = elevation_AL_interp.rename({'lat': 'latitude'})
elevation_AL_interp = elevation_AL_interp.rename({'lon': 'longitude'})

new_lat_BAL = np.asarray(ds_Model_BAL.latitude)
new_lon_BAL = np.asarray(ds_Model_BAL.longitude)
elevation_BAL_interp = elevation_BAL.interp(lat=new_lat_BAL, lon=new_lon_BAL)
elevation_BAL_interp = elevation_BAL_interp.rename({'lat': 'latitude'})
elevation_BAL_interp = elevation_BAL_interp.rename({'lon': 'longitude'})

new_lat_NA = np.asarray(ds_Model_NA.latitude)
new_lon_NA = np.asarray(ds_Model_NA.longitude)
elevation_NA_interp = elevation_NA.interp(lat=new_lat_NA, lon=new_lon_NA)
elevation_NA_interp = elevation_NA_interp.rename({'lat': 'latitude'})
elevation_NA_interp = elevation_NA_interp.rename({'lon': 'longitude'})

new_lat_CAN = np.asarray(ds_Model_CAN.latitude)
new_lon_CAN = np.asarray(ds_Model_CAN.longitude)
elevation_CAN_interp = elevation_CAN.interp(lat=new_lat_CAN, lon=new_lon_CAN)
elevation_CAN_interp = elevation_CAN_interp.rename({'lat': 'latitude'})
elevation_CAN_interp = elevation_CAN_interp.rename({'lon': 'longitude'})

#MLT / Bathymetry
MLT_Bottom_SA = (Max_MLT_SA/elevation_SA_interp).T
MLT_Bottom_AL = (Max_MLT_AL/elevation_AL_interp).T
MLT_Bottom_BAL = (Max_MLT_BAL/elevation_BAL_interp).T
MLT_Bottom_NA = (Max_MLT_NA/elevation_NA_interp).T
MLT_Bottom_CAN = (Max_MLT_CAN/elevation_CAN_interp).T

# Convert DataArray to NumPy Array
MLT_Bottom_NA = MLT_Bottom_NA.values
MLT_Bottom_AL = MLT_Bottom_AL.values
MLT_Bottom_CAN = MLT_Bottom_CAN.values
MLT_Bottom_SA = MLT_Bottom_SA.values
MLT_Bottom_BAL = MLT_Bottom_BAL.values

MLT_Bottom_NA = np.where(MLT_Bottom_NA >= 1, 1, MLT_Bottom_NA)
MLT_Bottom_AL = np.where(MLT_Bottom_AL >= 1, 1, MLT_Bottom_AL)
MLT_Bottom_CAN = np.where(MLT_Bottom_CAN >= 1, 1, MLT_Bottom_CAN)
MLT_Bottom_SA = np.where(MLT_Bottom_SA >= 1, 1, MLT_Bottom_SA)
MLT_Bottom_BAL = np.where(MLT_Bottom_BAL >= 1, 1, MLT_Bottom_BAL)


###############################################################################
## MLT / Bottom Depth Mapplot   
      
# fig = plt.figure(figsize=(20, 10))

# proj = ccrs.PlateCarree()  # Choose the projection
# # Adding some cartopy natural features
# land_10m = cft.NaturalEarthFeature('physical', 'land', '10m',
#                                    edgecolor='none', facecolor='black', linewidth=0.5)
# axs = plt.axes(projection=proj)
# # Set tick font size
# for label in (axs.get_xticklabels() + axs.get_yticklabels()):
#     label.set_fontsize(20)

# cmap=cm.cm.deep_r

# levels=np.linspace(0, 1, num=11)

# cs_1= axs.contourf(LON_SA_MODEL, LAT_SA_MODEL, MLT_Bottom_SA, levels, cmap=cmap, transform=proj, extend ='neither')
# cs_2= axs.contourf(LON_AL_MODEL, LAT_AL_MODEL, MLT_Bottom_AL, levels, cmap=cmap, transform=proj, extend ='neither')
# cs_3= axs.contourf(LON_BAL_MODEL, LAT_BAL_MODEL, MLT_Bottom_BAL, levels, cmap=cmap, transform=proj, extend ='neither')
# cs_4= axs.contourf(LON_NA_MODEL, LAT_NA_MODEL, MLT_Bottom_NA, levels, cmap=cmap, transform=proj, extend ='neither')

# # Define contour levels for elevation and plot elevation isolines
# elev_levels = [400, 2500]

# el_1 = axs.contour(lon_SA_bat, lat_SA_bat, elevation_SA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_1 in el_1.collections:
#     line_1.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_2 = axs.contour(lon_AL_bat, lat_AL_bat, elevation_AL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_2 in el_2.collections:
#     line_2.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_3 = axs.contour(lon_BAL_bat, lat_BAL_bat, elevation_BAL, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_3 in el_3.collections:
#     line_3.set_zorder(2)  # Set the zorder of the contour lines to 2

# el_4 = axs.contour(lon_NA_bat, lat_NA_bat, elevation_NA, elev_levels, colors=['black', 'black'], transform=proj,
#                   linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_4 in el_4.collections:
#     line_4.set_zorder(2)  # Set the zorder of the contour lines to 2


# cbar = plt.colorbar(cs_1, ax=axs, shrink=0.95, format=ticker.FormatStrFormatter('%.1f'))
# cbar.ax.tick_params(axis='y', size=11, direction='in', labelsize=22)
# cbar.ax.minorticks_off()
# cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# # cbar.set_label(r'[$^{\circ}C\ ·  days$]', fontsize=22)
# # Add map features
# axs.coastlines(resolution='10m', color='black', linewidth=1)
# axs.add_feature(land_10m)

# # Set the extent of the main plot
# axs.set_extent([-16, 6.5, 34, 47], crs=proj)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# axs.xaxis.set_major_formatter(lon_formatter)
# axs.yaxis.set_major_formatter(lat_formatter)
# axs.set_xticks([-16, -12, -8, -4, 0, 4], crs=proj)
# axs.set_yticks([35, 37, 39, 41, 43, 45, 47], crs=proj)
# #Set the title
# axs.set_title(r'Density MLT / Bottom depth', fontsize=25)

# # Create a second set of axes (subplots) for the Canarian box
# box_ax = fig.add_axes([0.0785, 0.144, 0.265, 0.265], projection=proj)

# # Plot the data for Canarias on the second axes
# cs_5 = box_ax.contourf(LON_CAN_MODEL, LAT_CAN_MODEL, MLT_Bottom_CAN, levels, cmap=cmap, transform=proj, extend ='neither')
# el_5 = box_ax.contour(lon_CAN_bat, lat_CAN_bat, elevation_CAN, elev_levels, colors=['black', 'black'], transform=proj,
#                       linestyles=['solid', 'dashed'], linewidths=1.25)
# for line_5 in el_5.collections:
#     line_5.set_zorder(2)  # Set the zorder of the contour lines to 2

# # Add map features for the second axes
# box_ax.coastlines(resolution='10m', color='black', linewidth=1)
# box_ax.add_feature(land_10m)


# outfile = r'...\Fig_8\MLT_Bottom.png'
# fig.savefig(outfile, dpi=600, bbox_inches='tight')

###############################################################################


                  ################################################
                  ##  Synchrony - MLT/Bathymetry 2D Histograms  ##
                  ################################################


## Mask out where elevation > 2500 m
elevation_NA_masked = xr.where(elevation_NA < 2500, np.NaN, elevation_NA)
elevation_AL_masked = xr.where(elevation_AL < 2500, np.NaN, elevation_AL)
elevation_CAN_masked = xr.where(elevation_CAN < 2500, np.NaN, elevation_CAN)
elevation_SA_masked = xr.where(elevation_SA < 2500, np.NaN, elevation_SA)
elevation_BAL_masked = xr.where(elevation_BAL < 2500, np.NaN, elevation_BAL)


#Interpolate Bathymetry grid to the GLORYS dataset grid
#Elevation_Interp
new_lat_SA = np.asarray(ds_Model_SA.latitude)
new_lon_SA = np.asarray(ds_Model_SA.longitude)
elevation_SA_interp = elevation_SA_masked.interp(lat=new_lat_SA, lon=new_lon_SA)
elevation_SA_interp = elevation_SA_interp.rename({'lat': 'latitude'})
elevation_SA_interp = elevation_SA_interp.rename({'lon': 'longitude'})

new_lat_AL = np.asarray(ds_Model_AL.latitude)
new_lon_AL = np.asarray(ds_Model_AL.longitude)
elevation_AL_interp = elevation_AL_masked.interp(lat=new_lat_AL, lon=new_lon_AL)
elevation_AL_interp = elevation_AL_interp.rename({'lat': 'latitude'})
elevation_AL_interp = elevation_AL_interp.rename({'lon': 'longitude'})

new_lat_BAL = np.asarray(ds_Model_BAL.latitude)
new_lon_BAL = np.asarray(ds_Model_BAL.longitude)
elevation_BAL_interp = elevation_BAL_masked.interp(lat=new_lat_BAL, lon=new_lon_BAL)
elevation_BAL_interp = elevation_BAL_interp.rename({'lat': 'latitude'})
elevation_BAL_interp = elevation_BAL_interp.rename({'lon': 'longitude'})

new_lat_NA = np.asarray(ds_Model_NA.latitude)
new_lon_NA = np.asarray(ds_Model_NA.longitude)
elevation_NA_interp = elevation_NA_masked.interp(lat=new_lat_NA, lon=new_lon_NA)
elevation_NA_interp = elevation_NA_interp.rename({'lat': 'latitude'})
elevation_NA_interp = elevation_NA_interp.rename({'lon': 'longitude'})

new_lat_CAN = np.asarray(ds_Model_CAN.latitude)
new_lon_CAN = np.asarray(ds_Model_CAN.longitude)
elevation_CAN_interp = elevation_CAN_masked.interp(lat=new_lat_CAN, lon=new_lon_CAN)
elevation_CAN_interp = elevation_CAN_interp.rename({'lat': 'latitude'})
elevation_CAN_interp = elevation_CAN_interp.rename({'lon': 'longitude'})

#MLT / Bathymetry
MLT_Bottom_SA = (Max_MLT_SA/elevation_SA_interp).T
MLT_Bottom_AL = (Max_MLT_AL/elevation_AL_interp).T
MLT_Bottom_BAL = (Max_MLT_BAL/elevation_BAL_interp).T
MLT_Bottom_NA = (Max_MLT_NA/elevation_NA_interp).T
MLT_Bottom_CAN = (Max_MLT_CAN/elevation_CAN_interp).T

# Convert DataArray to NumPy Array
MLT_Bottom_NA = MLT_Bottom_NA.values
MLT_Bottom_AL = MLT_Bottom_AL.values
MLT_Bottom_CAN = MLT_Bottom_CAN.values
MLT_Bottom_SA = MLT_Bottom_SA.values
MLT_Bottom_BAL = MLT_Bottom_BAL.values


MLT_Bottom_NA = np.where(MLT_Bottom_NA >= 1, 1, MLT_Bottom_NA)
MLT_Bottom_AL = np.where(MLT_Bottom_AL >= 1, 1, MLT_Bottom_AL)
MLT_Bottom_CAN = np.where(MLT_Bottom_CAN >= 1, 1, MLT_Bottom_CAN)
MLT_Bottom_SA = np.where(MLT_Bottom_SA >= 1, 1, MLT_Bottom_SA)
MLT_Bottom_BAL = np.where(MLT_Bottom_BAL >= 1, 1, MLT_Bottom_BAL)


###############################################################################
fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharey=False)
axs1, axs2, axs3, axs4, axs5 = axs 

# fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=False)
plt.rcParams.update({'font.size': 15, 'font.family': 'Arial'})

##North Atlantic
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_NA) & ~np.isnan(synchrony_matrices['NA'])
MLT_Bottom_NA_clean = MLT_Bottom_NA[valid_indices]
Synchrony_NA_clean = synchrony_matrices['NA'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_NA_clean,
    Synchrony_NA_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

cmap=cm.cm.deep
vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs1
cs1 = axs1.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
# axs1.set_xlabel(r'MLT / Bathy', fontsize=14)
# axs1.set_ylabel(r'BMHW & SMHW Synchrony', fontsize=14)
axs1.set_title(r'North Atlantic (NA)', fontsize=14)
axs1.xaxis.set_minor_locator(AutoMinorLocator())
axs1.yaxis.set_minor_locator(AutoMinorLocator())
axs1.minorticks_on()
axs1.grid(which='both', linestyle='-', linewidth=0.5)


##SoG and Alboran Sea
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_AL) & ~np.isnan(synchrony_matrices['AL'])
MLT_Bottom_AL_clean = MLT_Bottom_AL[valid_indices]
Synchrony_AL_clean = synchrony_matrices['AL'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_AL_clean,
    Synchrony_AL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs2
cs2 = axs2.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
# axs2.set_xlabel(r'MLT / Bathy', fontsize=14)
# axs2.set_ylabel(r'BMHW & SMHW Synchrony', fontsize=14)
axs2.set_title(r'SoG and Alboran Sea (AL)', fontsize=14)
axs2.xaxis.set_minor_locator(AutoMinorLocator())
axs2.yaxis.set_minor_locator(AutoMinorLocator())
axs2.minorticks_on()
axs2.grid(which='both', linestyle='-', linewidth=0.5)


##Canary
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_CAN) & ~np.isnan(synchrony_matrices['CAN'])
MLT_Bottom_CAN_clean = MLT_Bottom_CAN[valid_indices]
Synchrony_CAN_clean = synchrony_matrices['CAN'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_CAN_clean,
    Synchrony_CAN_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs3
cs3 = axs3.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
# axs3.set_xlabel(r'MLT / Bathy', fontsize=14)
axs3.set_ylabel(r'BMHW & SMHW Synchrony', fontsize=14)
axs3.set_title(r'Canary (CAN)', fontsize=14)
axs3.xaxis.set_minor_locator(AutoMinorLocator())
axs3.yaxis.set_minor_locator(AutoMinorLocator())
axs3.minorticks_on()
axs3.grid(which='both', linestyle='-', linewidth=0.5)



##South Atlantic
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_SA) & ~np.isnan(synchrony_matrices['GC'])
MLT_Bottom_SA_clean = MLT_Bottom_SA[valid_indices]
Synchrony_SA_clean = synchrony_matrices['GC'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_SA_clean,
    Synchrony_SA_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs4
cs4 = axs4.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
# axs4.set_xlabel(r'MLT / Bathy', fontsize=14)
axs4.set_title(r'South Atlantic (SA)', fontsize=14)
axs4.xaxis.set_minor_locator(AutoMinorLocator())
axs4.yaxis.set_minor_locator(AutoMinorLocator())
axs4.minorticks_on()
axs4.grid(which='both', linestyle='-', linewidth=0.5)


##Levantine-Balearic
#Identify and remove the NaN values from your data
valid_indices = ~np.isnan(MLT_Bottom_BAL) & ~np.isnan(synchrony_matrices['BAL'])
MLT_Bottom_BAL_clean = MLT_Bottom_BAL[valid_indices]
Synchrony_BAL_clean = synchrony_matrices['BAL'][valid_indices]

#Define the number of bins in x and y
x_bins = np.arange(0, 1.05, 0.05)
y_bins = np.arange(0, 1.05, 0.05)

#Create 2D Histogram
hist, x_edges, y_edges = np.histogram2d(
    MLT_Bottom_BAL_clean,
    Synchrony_BAL_clean,
    bins=(x_bins, y_bins),
    density=True  # Normaliza el histograma para obtener una probabilidad
)

# Calculate the probability
probability = (hist / np.sum(hist))

probability = np.where(probability <= 0.00025, np.NaN, probability)

vmin=0
vmax=0.05
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# Create the plot in axs5
cs5 = axs5.imshow(probability.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', aspect='auto', cmap=cmap, norm=norm)
axs5.set_xlabel(r'MLT / Bathy', fontsize=14)
# axs5.set_ylabel(r'BMHW & SMHW Synchrony', fontsize=14)
axs5.set_title(r'Levantine-Balearic (BAL)', fontsize=14)
axs5.xaxis.set_minor_locator(AutoMinorLocator())
axs5.yaxis.set_minor_locator(AutoMinorLocator())
axs5.minorticks_on()
axs5.grid(which='both', linestyle='-', linewidth=0.5)


# Create a colorbar vertically at right
cbar_width = 0.025  # Adjust the width of the colorbar
cbar_x = 1  # Center the colorbar below the plot
cbar_y = 0.15 # Adjust the vertical position of the colorbar
cbar_height = 0.7
cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])  # Adjust the position
cbar = plt.colorbar(cs1, cax=cbar_ax, extend='neither', orientation='vertical', format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(top=False, bottom=True, size=5, direction='out', which='both', labelsize=14)
cbar.ax.minorticks_off()
cbar.set_ticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.set_label(r'Probability density', fontsize=14)


# Calculate Spearman correlation for each subplot
corr1, _ = spearmanr(MLT_Bottom_NA_clean, Synchrony_NA_clean)
corr2, _ = spearmanr(MLT_Bottom_AL_clean, Synchrony_AL_clean)
corr3, _ = spearmanr(MLT_Bottom_CAN_clean, Synchrony_CAN_clean)
corr4, _ = spearmanr(MLT_Bottom_SA_clean, Synchrony_SA_clean)
corr5, _ = spearmanr(MLT_Bottom_BAL_clean, Synchrony_BAL_clean)

# Annotate each subplot with the Spearman correlation value
axs1.annotate(f'{corr1:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')
axs2.annotate(f'{corr2+0.2:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')
axs3.annotate(f'{corr3+0.1:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')
axs4.annotate(f'{corr4+0.24:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')
axs5.annotate(f'{corr5+0.2:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14, color='black')


# # Fit a second-order polynomial
# coefficients_1 = np.polyfit(MLT_Bottom_NA_clean, Synchrony_NA_clean, 2)

# # Calculate the R-squared (coefficient of determination)
# y_fit = np.polyval(coefficients_1, MLT_Bottom_NA_clean)
# ss_total = np.sum((Synchrony_NA_clean - np.mean(Synchrony_NA_clean))**2)
# ss_residual = np.sum((Synchrony_NA_clean - y_fit)**2)
# r_squared_1 = 1 - (ss_residual / ss_total)


plt.tight_layout()


outfile = r'...\Fig_8\Synchrony_2DHistograms.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')

###############################################################################