# -*- coding: utf-8 -*-
"""

#########################      Figure 5 in 
Fernández-Barba, M., Huertas, I. E., & Navarro, G. (2024). 
Assessment of surface and bottom marine heatwaves along the Spanish coast. 
Ocean Modelling, 190, 102399.                          ########################

"""

#Loading required python modules
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as colors
import matplotlib.ticker as ticker


##Load MHWs_from_MATLAB.py


### SPATIAL EXTENT OF SMHW & BMHW PROPERTIES ###

time = np.arange(1993, 2023)

def calculate_fraction_area(array, threshold):
    # Calcula la fracción de área para la cual el valor de MHW es igual o mayor al umbral dado
    freq_greater_than_threshold = np.sum(array >= threshold, axis=(0, 1))
    total_area = array.shape[0] * array.shape[1]
    fraction_area = freq_greater_than_threshold / total_area
    return fraction_area



## Total Annual MHW Days

#SMHWs
Td_NA_ts = np.nanmean(MHW_td_ts_NA_MODEL, axis=(0, 1))
Td_AL_ts = np.nanmean(MHW_td_ts_AL_MODEL, axis=(0, 1))
Td_CAN_ts = np.nanmean(MHW_td_ts_CAN_MODEL, axis=(0, 1))
Td_SA_ts = np.nanmean(MHW_td_ts_SA_MODEL, axis=(0, 1))
Td_BAL_ts = np.nanmean(MHW_td_ts_BAL_MODEL, axis=(0, 1))
#BMHWs
Bottom_Td_NA_ts = np.nanmean(BMHW_td_ts_NA_MODEL, axis=(0, 1))
Bottom_Td_AL_ts = np.nanmean(BMHW_td_ts_AL_MODEL, axis=(0, 1))
Bottom_Td_CAN_ts = np.nanmean(BMHW_td_ts_CAN_MODEL, axis=(0, 1))
Bottom_Td_SA_ts = np.nanmean(BMHW_td_ts_SA_MODEL, axis=(0, 1))
Bottom_Td_BAL_ts = np.nanmean(BMHW_td_ts_BAL_MODEL, axis=(0, 1))

#Replacing NaNs with 0s
Td_NA_ts = np.nan_to_num(Td_NA_ts, nan=0.0)
Td_AL_ts = np.nan_to_num(Td_AL_ts, nan=0.0)
Td_CAN_ts = np.nan_to_num(Td_CAN_ts, nan=0.0)
Td_SA_ts = np.nan_to_num(Td_SA_ts, nan=0.0)
Td_BAL_ts = np.nan_to_num(Td_BAL_ts, nan=0.0)


#Calculating errors
error_Td_NA = (np.nanstd(MHW_td_ts_NA_MODEL, axis=(0, 1)) / np.sqrt(30)) 
error_Td_AL = (np.nanstd(MHW_td_ts_AL_MODEL, axis=(0, 1)) / np.sqrt(30)) 
error_Td_CAN = (np.nanstd(MHW_td_ts_CAN_MODEL, axis=(0, 1)) / np.sqrt(30)) 
error_Td_SA = (np.nanstd(MHW_td_ts_SA_MODEL, axis=(0, 1)) / np.sqrt(30)) 
error_Td_BAL = (np.nanstd(MHW_td_ts_BAL_MODEL, axis=(0, 1)) / np.sqrt(30))

error_bottom_Td_NA = (np.nanstd(BMHW_td_ts_NA_MODEL, axis=(0, 1)) / np.sqrt(30)) 
error_bottom_Td_AL = (np.nanstd(BMHW_td_ts_AL_MODEL, axis=(0, 1)) / np.sqrt(30)) 
error_bottom_Td_CAN = (np.nanstd(BMHW_td_ts_CAN_MODEL, axis=(0, 1)) / np.sqrt(30)) 
error_bottom_Td_SA = (np.nanstd(BMHW_td_ts_SA_MODEL, axis=(0, 1)) / np.sqrt(30)) 
error_bottom_Td_BAL = (np.nanstd(BMHW_td_ts_BAL_MODEL, axis=(0, 1)) / np.sqrt(30))


# Normalize each error value between 0 and 1
def normalize_error(error, min_range, max_range):
    min_value = np.nanmin(error)
    max_value = np.nanmax(error)
    normalized_error = min_range + (max_range - min_range) * ((error - min_value) / (max_value - min_value))
    return normalized_error

# Set the desired range for error normalization
min_error_range = 0
max_error_range = 0.25


normalized_error_Td_CAN = normalize_error(error_Td_CAN, min_error_range, max_error_range) 
normalized_error_Td_SA = normalize_error(error_Td_SA, min_error_range, max_error_range) 
normalized_error_Td_AL = normalize_error(error_Td_AL, min_error_range, max_error_range) 
normalized_error_Td_BAL = normalize_error(error_Td_BAL, min_error_range, max_error_range) 
normalized_error_Td_NA = normalize_error(error_Td_NA, min_error_range, max_error_range) 

normalized_error_bottom_Td_CAN = normalize_error(error_bottom_Td_CAN, min_error_range, max_error_range) 
normalized_error_bottom_Td_SA = normalize_error(error_bottom_Td_SA, min_error_range, max_error_range) 
normalized_error_bottom_Td_AL = normalize_error(error_bottom_Td_AL, min_error_range, max_error_range) 
normalized_error_bottom_Td_BAL = normalize_error(error_bottom_Td_BAL, min_error_range, max_error_range) 
normalized_error_bottom_Td_NA = normalize_error(error_bottom_Td_NA, min_error_range, max_error_range) 

#Calculating MHW Fraction of Area (FoA)
threshold = 5
fraction_area_NA = calculate_fraction_area(MHW_td_ts_NA_MODEL, threshold)
fraction_area_AL = calculate_fraction_area(MHW_td_ts_AL_MODEL, threshold)
fraction_area_CAN = calculate_fraction_area(MHW_td_ts_CAN_MODEL, threshold)
fraction_area_SA = calculate_fraction_area(MHW_td_ts_SA_MODEL, threshold)
fraction_area_BAL = calculate_fraction_area(MHW_td_ts_BAL_MODEL, threshold)

Bottom_fraction_area_NA = calculate_fraction_area(BMHW_td_ts_NA_MODEL, threshold)
Bottom_fraction_area_AL = calculate_fraction_area(BMHW_td_ts_AL_MODEL, threshold)
Bottom_fraction_area_CAN = calculate_fraction_area(BMHW_td_ts_CAN_MODEL, threshold)
Bottom_fraction_area_SA = calculate_fraction_area(BMHW_td_ts_SA_MODEL, threshold)
Bottom_fraction_area_BAL = calculate_fraction_area(BMHW_td_ts_BAL_MODEL, threshold)

#Normalizing FoA
Max_fraction_area_NA = np.nanmax(fraction_area_NA, axis=0)
Max_fraction_area_AL = np.nanmax(fraction_area_AL, axis=0)
Max_fraction_area_CAN = np.nanmax(fraction_area_CAN, axis=0)
Max_fraction_area_SA = np.nanmax(fraction_area_SA, axis=0)
Max_fraction_area_BAL = np.nanmax(fraction_area_BAL, axis=0)

Bottom_td_fraction_area_NA = np.nanmax(Bottom_fraction_area_NA, axis=0)
Bottom_td_fraction_area_AL = np.nanmax(Bottom_fraction_area_AL, axis=0)
Bottom_td_fraction_area_CAN = np.nanmax(Bottom_fraction_area_CAN, axis=0)
Bottom_td_fraction_area_SA = np.nanmax(Bottom_fraction_area_SA, axis=0)
Bottom_td_fraction_area_BAL = np.nanmax(Bottom_fraction_area_BAL, axis=0)

norm_fraction_area_NA = fraction_area_NA / Max_fraction_area_NA
norm_fraction_area_AL = fraction_area_AL / Max_fraction_area_AL
norm_fraction_area_CAN = fraction_area_CAN / Max_fraction_area_CAN
norm_fraction_area_SA = fraction_area_SA / Max_fraction_area_SA
norm_fraction_area_BAL = fraction_area_BAL / Max_fraction_area_BAL

Bottom_norm_fraction_area_NA = Bottom_fraction_area_NA / Bottom_td_fraction_area_NA
Bottom_norm_fraction_area_AL = Bottom_fraction_area_AL / Bottom_td_fraction_area_AL
Bottom_norm_fraction_area_CAN = Bottom_fraction_area_CAN / Bottom_td_fraction_area_CAN
Bottom_norm_fraction_area_SA = Bottom_fraction_area_SA / Bottom_td_fraction_area_SA
Bottom_norm_fraction_area_BAL = Bottom_fraction_area_BAL / Bottom_td_fraction_area_BAL


#Ensure that the fraction of area is between 0 and 1
norm_fraction_area_NA = np.clip(norm_fraction_area_NA, 0, 1)
norm_fraction_area_AL = np.clip(norm_fraction_area_AL, 0, 1)
norm_fraction_area_CAN = np.clip(norm_fraction_area_CAN, 0, 1)
norm_fraction_area_SA = np.clip(norm_fraction_area_SA, 0, 1)
norm_fraction_area_BAL = np.clip(norm_fraction_area_BAL, 0, 1)



#Plotting the figure
fig, (axs1, axs2, axs3, axs4, axs5) = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})
norm = colors.Normalize(vmin=5, vmax=40)
norm_bottom = colors.Normalize(vmin=5, vmax=40)
cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlOrRd)
# cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlBu_r)
cmap_bottom = plt.cm.ScalarMappable(norm=norm_bottom, cmap=plt.cm.YlOrRd)

#North Atlantic
axs1.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding normalized bars
cs1 = axs1.bar(time, norm_fraction_area_NA, color=cmap.to_rgba(Td_NA_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs1):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_NA[i]
    axs1.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css1 = axs1.scatter(time, Bottom_norm_fraction_area_NA, color=cmap_bottom.to_rgba(Bottom_Td_NA_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css1.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_NA[i]
    axs1.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)

axs1.set_xlim(1992, 2023)
axs1.set_ylim(0, 1, 0.25)
axs1.set_yticks([0, 0.5, 1])
axs1.tick_params(length=5, direction='out', which='both', right=True)
axs1.yaxis.set_label_position('right')
axs1.text(0.02, 1.18, 'a', transform=axs1.transAxes, ha='left', va='top', fontsize=14, weight ='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs1.set_title(r'North Atlantic (NA)', fontsize=14)


#Strait of Gibraltar and Alboran Sea
axs2.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs2 = axs2.bar(time, norm_fraction_area_AL, color=cmap.to_rgba(Td_AL_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs2):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_AL[i]
    axs2.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css2= axs2.scatter(time, Bottom_norm_fraction_area_AL, color=cmap_bottom.to_rgba(Bottom_Td_AL_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css2.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_AL[i]
    axs2.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)

axs2.set_xlim(1992, 2023)
axs2.set_ylim(0, 1, 0.25)
axs2.set_yticks([0, 0.5, 1])
axs2.tick_params(length=5, direction='out', which='both', right=True)
axs2.yaxis.set_label_position('right')
axs2.text(0.02, 1.18, 'd', transform=axs2.transAxes, ha='left', va='top', fontsize=14, weight ='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs2.set_title(r'SoG and Alboran Sea (AL)', fontsize=14)


#Canary
axs3.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs3 = axs3.bar(time, norm_fraction_area_CAN, color=cmap.to_rgba(Td_CAN_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs3):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_CAN[i]
    axs3.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css3 = axs3.scatter(time, Bottom_norm_fraction_area_CAN, color=cmap_bottom.to_rgba(Bottom_Td_CAN_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css3.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_CAN[i]
    axs3.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)
    
axs3.set_xlim(1992, 2023)
axs3.set_ylim(0, 1, 0.25)
axs3.set_yticks([0, 0.5, 1])
axs3.tick_params(length=5, direction='out', which='both', right=True)
axs3.yaxis.set_label_position('right')
axs3.text(0.02, 1.18, 'g', transform=axs3.transAxes, ha='left', va='top', fontsize=14, weight ='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs3.set_title(r'Canary (CAN)', fontsize=14)


#South Atlantic
axs4.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs4 = axs4.bar(time, norm_fraction_area_SA, color=cmap.to_rgba(Td_SA_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs4):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_SA[i]
    axs4.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css4 = axs4.scatter(time, Bottom_norm_fraction_area_SA, color=cmap_bottom.to_rgba(Bottom_Td_SA_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css4.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_SA[i]
    axs4.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)
    
axs4.set_xlim(1992, 2023)
axs4.set_ylim(0, 1, 0.25)
axs4.set_yticks([0, 0.5, 1])
axs4.tick_params(length=5, direction='out', which='both', right=True)
axs4.yaxis.set_label_position('right')
axs4.text(0.02, 1.18, 'j', transform=axs4.transAxes, ha='left', va='top', fontsize=14, weight ='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs4.set_title(r'South Atlantic (SA)', fontsize=14)


#Levantine-Balearic
axs5.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs5 = axs5.bar(time, norm_fraction_area_BAL, color=cmap.to_rgba(Td_BAL_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs5):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_BAL[i]
    axs5.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css5 = axs5.scatter(time, Bottom_norm_fraction_area_BAL, color=cmap_bottom.to_rgba(Bottom_Td_BAL_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css5.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_BAL[i]
    axs5.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)
    
axs5.set_xlim(1992, 2023)
axs5.set_ylim(0, 1, 0.25)
axs5.set_yticks([0, 0.5, 1])
axs5.tick_params(length=5, direction='out', which='both', right=True)
axs5.yaxis.set_label_position('right')
axs5.text(0.02, 1.18, 'm', transform=axs5.transAxes, ha='left', va='top', fontsize=14, weight ='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs5.set_title(r'Levantine-Balearic (BAL)', fontsize=14)

# Create common colorbar
# Create colorbar for SMHW
cbar_ax = fig.add_axes([0.15, 0.023, 0.62, 0.015])
cbar = plt.colorbar(cmap, cax=cbar_ax, extend='max', orientation='horizontal', format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(top=True, bottom=False, size=5, direction='out', which='both', labelsize=14)
cbar.ax.minorticks_off()
cbar.set_ticks([5, 10, 15, 20, 25, 30, 35, 40])
cbar.ax.xaxis.set_ticks_position('top')
cbar.set_label(r'Averaged Total Annual SMHW Days', fontsize=14, labelpad=-50)

# Create colorbar for BMHW
cbar_bottom_ax = fig.add_axes([0.15, 0.023, 0.62, 0.015])  # Adjust the position
cbar_bottom = plt.colorbar(cmap_bottom, cax=cbar_bottom_ax, extend='max', orientation='horizontal', format=ticker.FormatStrFormatter('%.0f'))
cbar_bottom.ax.tick_params(top=False, bottom=True, size=5, direction='out', which='both', labelsize=14)
cbar_bottom.ax.minorticks_off()
cbar_bottom.set_ticks([5, 10, 15, 20, 25, 30, 35, 40])  # Define los ticks personalizados para cmap_bottom
cbar_bottom.ax.xaxis.set_ticks_position('bottom')
cbar_bottom.set_label(r'Averaged Total Annual BMHW Days', fontsize=14, labelpad=0)


fig.text(0.054, 0.5, 'Fraction of Area', va='center', rotation='vertical', fontsize=14)  # Agrega el título del eje y en el centro

plt.subplots_adjust(right=0.77, hspace=0.3)


outfile = r'...\Fig_5\Spatial_Extent_MHW_Td.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




## Maximum Intensity

#SMHWs
Max_CAN_ts = np.nanmean(MHW_max_ts_CAN_MODEL, axis=(0, 1))
Max_SA_ts = np.nanmean(MHW_max_ts_SA_MODEL, axis=(0, 1))
Max_AL_ts = np.nanmean(MHW_max_ts_AL_MODEL, axis=(0, 1))
Max_BAL_ts = np.nanmean(MHW_max_ts_BAL_MODEL, axis=(0, 1))
Max_NA_ts = np.nanmean(MHW_max_ts_NA_MODEL, axis=(0, 1))
#BMHWs
Bottom_Max_CAN_ts = np.nanmean(BMHW_max_ts_CAN_MODEL, axis=(0, 1))
Bottom_Max_SA_ts = np.nanmean(BMHW_max_ts_SA_MODEL, axis=(0, 1))
Bottom_Max_AL_ts = np.nanmean(BMHW_max_ts_AL_MODEL, axis=(0, 1))
Bottom_Max_BAL_ts = np.nanmean(BMHW_max_ts_BAL_MODEL, axis=(0, 1))
Bottom_Max_NA_ts = np.nanmean(BMHW_max_ts_NA_MODEL, axis=(0, 1))

#Replacing NaNs with 0s
Max_CAN_ts = np.nan_to_num(Max_CAN_ts, nan=0.0)
Max_SA_ts = np.nan_to_num(Max_SA_ts, nan=0.0)
Max_AL_ts = np.nan_to_num(Max_AL_ts, nan=0.0)
Max_BAL_ts = np.nan_to_num(Max_BAL_ts, nan=0.0)
Max_NA_ts = np.nan_to_num(Max_NA_ts, nan=0.0)

#Normalizing metric
max_Max_CAN = np.nanmax(MHW_max_ts_CAN_MODEL, axis=(0, 1))
max_Max_SA = np.nanmax(MHW_max_ts_SA_MODEL, axis=(0, 1))
max_Max_AL = np.nanmax(MHW_max_ts_AL_MODEL, axis=(0, 1))
max_Max_BAL = np.nanmax(MHW_max_ts_BAL_MODEL, axis=(0, 1))
max_Max_NA = np.nanmax(MHW_max_ts_NA_MODEL, axis=(0, 1))


#Calculating MHW Fraction of Area (FoA)
threshold = 0.01
fraction_area_NA = calculate_fraction_area(MHW_max_ts_NA_MODEL, threshold)
fraction_area_AL = calculate_fraction_area(MHW_max_ts_AL_MODEL, threshold)
fraction_area_CAN = calculate_fraction_area(MHW_max_ts_CAN_MODEL, threshold)
fraction_area_SA = calculate_fraction_area(MHW_max_ts_SA_MODEL, threshold)
fraction_area_BAL = calculate_fraction_area(MHW_max_ts_BAL_MODEL, threshold)

Bottom_fraction_area_NA = calculate_fraction_area(BMHW_max_ts_NA_MODEL, threshold)
Bottom_fraction_area_AL = calculate_fraction_area(BMHW_max_ts_AL_MODEL, threshold)
Bottom_fraction_area_CAN = calculate_fraction_area(BMHW_max_ts_CAN_MODEL, threshold)
Bottom_fraction_area_SA = calculate_fraction_area(BMHW_max_ts_SA_MODEL, threshold)
Bottom_fraction_area_BAL = calculate_fraction_area(BMHW_max_ts_BAL_MODEL, threshold)

#Normalizing FoA
Max_fraction_area_NA = np.nanmax(fraction_area_NA, axis=0)
Max_fraction_area_AL = np.nanmax(fraction_area_AL, axis=0)
Max_fraction_area_CAN = np.nanmax(fraction_area_CAN, axis=0)
Max_fraction_area_SA = np.nanmax(fraction_area_SA, axis=0)
Max_fraction_area_BAL = np.nanmax(fraction_area_BAL, axis=0)

Bottom_max_fraction_area_NA = np.nanmax(Bottom_fraction_area_NA, axis=0)
Bottom_max_fraction_area_AL = np.nanmax(Bottom_fraction_area_AL, axis=0)
Bottom_max_fraction_area_CAN = np.nanmax(Bottom_fraction_area_CAN, axis=0)
Bottom_max_fraction_area_SA = np.nanmax(Bottom_fraction_area_SA, axis=0)
Bottom_max_fraction_area_BAL = np.nanmax(Bottom_fraction_area_BAL, axis=0)

norm_fraction_area_NA = fraction_area_NA / Max_fraction_area_NA
norm_fraction_area_AL = fraction_area_AL / Max_fraction_area_AL
norm_fraction_area_CAN = fraction_area_CAN / Max_fraction_area_CAN
norm_fraction_area_SA = fraction_area_SA / Max_fraction_area_SA
norm_fraction_area_BAL = fraction_area_BAL / Max_fraction_area_BAL

Bottom_norm_fraction_area_NA = Bottom_fraction_area_NA / Bottom_max_fraction_area_NA
Bottom_norm_fraction_area_AL = Bottom_fraction_area_AL / Bottom_max_fraction_area_AL
Bottom_norm_fraction_area_CAN = Bottom_fraction_area_CAN / Bottom_max_fraction_area_CAN
Bottom_norm_fraction_area_SA = Bottom_fraction_area_SA / Bottom_max_fraction_area_SA
Bottom_norm_fraction_area_BAL = Bottom_fraction_area_BAL / Bottom_max_fraction_area_BAL


#Ensure that the fraction of area is between 0 and 1
norm_fraction_area_NA = np.clip(norm_fraction_area_NA, 0, 1)
norm_fraction_area_AL = np.clip(norm_fraction_area_AL, 0, 1)
norm_fraction_area_CAN = np.clip(norm_fraction_area_CAN, 0, 1)
norm_fraction_area_SA = np.clip(norm_fraction_area_SA, 0, 1)
norm_fraction_area_BAL = np.clip(norm_fraction_area_BAL, 0, 1)


#Plotting the figure
fig, (axs1, axs2, axs3, axs4, axs5) = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})
norm = colors.Normalize(vmin=0.75, vmax=3.25)
norm_bottom = colors.Normalize(vmin=0.1, vmax=1.85)
cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlOrRd)
# cmap = plt.cm.ScalarMappable(norm=norm, cmap=cm.cm.thermal)
cmap_bottom = plt.cm.ScalarMappable(norm=norm_bottom, cmap=plt.cm.YlOrRd)


#North Atlantic
axs1.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding normalized bars
cs1 = axs1.bar(time, norm_fraction_area_NA, color=cmap.to_rgba(Max_NA_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs1):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_NA[i]
    axs1.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css1 = axs1.scatter(time, Bottom_norm_fraction_area_NA, color=cmap_bottom.to_rgba(Bottom_Max_NA_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css1.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_NA[i]
    axs1.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)

axs1.set_xlim(1992, 2023)
axs1.set_ylim(0, 1, 0.25)
axs1.set_yticks([0, 0.5, 1])
axs1.tick_params(length=5, direction='out', which='both', right=True)
axs1.yaxis.set_label_position('right')
axs1.text(0.02, 1.18, 'b', transform=axs1.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs1.set_title(r'North Atlantic (NA)', fontsize=14)


#Strait of Gibraltar and Alboran Sea
axs2.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs2 = axs2.bar(time, norm_fraction_area_AL, color=cmap.to_rgba(Max_AL_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs2):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_AL[i]
    axs2.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css2= axs2.scatter(time, Bottom_norm_fraction_area_AL, color=cmap_bottom.to_rgba(Bottom_Max_AL_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css2.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_AL[i]
    axs2.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)

axs2.set_xlim(1992, 2023)
axs2.set_ylim(0, 1, 0.25)
axs2.set_yticks([0, 0.5, 1])
axs2.tick_params(length=5, direction='out', which='both', right=True)
axs2.yaxis.set_label_position('right')
axs2.text(0.02, 1.18, 'e', transform=axs2.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs2.set_title(r'SoG and Alboran Sea (AL)', fontsize=14)


#Canary
axs3.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs3 = axs3.bar(time, norm_fraction_area_CAN, color=cmap.to_rgba(Max_CAN_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs3):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_CAN[i]
    axs3.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css3 = axs3.scatter(time, Bottom_norm_fraction_area_CAN, color=cmap_bottom.to_rgba(Bottom_Max_CAN_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css3.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_CAN[i]
    axs3.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)
    
axs3.set_xlim(1992, 2023)
axs3.set_ylim(0, 1, 0.25)
axs3.set_yticks([0, 0.5, 1])
axs3.tick_params(length=5, direction='out', which='both', right=True)
axs3.yaxis.set_label_position('right')
axs3.text(0.02, 1.18, 'h', transform=axs3.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs3.set_title(r'Canary (CAN)', fontsize=14)


#South Atlantic
axs4.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs4 = axs4.bar(time, norm_fraction_area_SA, color=cmap.to_rgba(Max_SA_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs4):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_SA[i]
    axs4.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css4 = axs4.scatter(time, Bottom_norm_fraction_area_SA, color=cmap_bottom.to_rgba(Bottom_Max_SA_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css4.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_SA[i]
    axs4.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)
    
axs4.set_xlim(1992, 2023)
axs4.set_ylim(0, 1, 0.25)
axs4.set_yticks([0, 0.5, 1])
axs4.tick_params(length=5, direction='out', which='both', right=True)
axs4.yaxis.set_label_position('right')
axs4.text(0.02, 1.18, 'k', transform=axs4.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs4.set_title(r'South Atlantic (SA)', fontsize=14)


#Levantine-Balearic
axs5.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs5 = axs5.bar(time, norm_fraction_area_BAL, color=cmap.to_rgba(Max_BAL_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs5):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_BAL[i]
    axs5.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css5 = axs5.scatter(time, Bottom_norm_fraction_area_BAL, color=cmap_bottom.to_rgba(Bottom_Max_BAL_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css5.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_BAL[i]
    axs5.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)
    
axs5.set_xlim(1992, 2023)
axs5.set_ylim(0, 1, 0.25)
axs5.set_yticks([0, 0.5, 1])
axs5.tick_params(length=5, direction='out', which='both', right=True)
axs5.yaxis.set_label_position('right')
axs5.text(0.02, 1.18, 'n', transform=axs5.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs5.set_title(r'Levantine-Balearic (BAL)', fontsize=14)

# Create common colorbar
# Create colorbar for SMHW
cbar_ax = fig.add_axes([0.15, 0.023, 0.62, 0.015])
cbar = plt.colorbar(cmap, cax=cbar_ax, extend='max', orientation='horizontal', format=ticker.FormatStrFormatter('%.2f'))
cbar.ax.tick_params(top=True, bottom=False, size=5, direction='out', which='both', labelsize=14)
cbar.ax.minorticks_off()
cbar.set_ticks([0.75, 1.25, 1.75, 2.25, 2.75, 3.25])
cbar.ax.xaxis.set_ticks_position('top')
cbar.set_label(r'Averaged SMHW Maximum Intensity [$^\circ$C]', fontsize=14, labelpad=-50)

# Create colorbar for BMHW
cbar_bottom_ax = fig.add_axes([0.15, 0.023, 0.62, 0.015])  # Adjust the position
cbar_bottom = plt.colorbar(cmap_bottom, cax=cbar_bottom_ax, extend='max', orientation='horizontal', format=ticker.FormatStrFormatter('%.2f'))
cbar_bottom.ax.tick_params(top=False, bottom=True, size=5, direction='out', which='both', labelsize=14)
cbar_bottom.ax.minorticks_off()
cbar_bottom.set_ticks([0.1, 0.45, 0.8, 1.15, 1.5, 1.85])
cbar_bottom.ax.xaxis.set_ticks_position('bottom')
cbar_bottom.set_label(r'Averaged BMHW Maximum Intensity [$^\circ$C]', fontsize=14, labelpad=0)

fig.text(0.054, 0.5, 'Fraction of Area', va='center', rotation='vertical', fontsize=14)  # Agrega el título del eje y en el centro

plt.subplots_adjust(right=0.77, hspace=0.3)


outfile = r'...\Fig_5\Spatial_Extent_MHW_MaxInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')




## Cumulative Intensity

#SMHWs
Cum_CAN_ts = np.nanmean(MHW_cum_ts_CAN_MODEL, axis=(0, 1))
Cum_SA_ts = np.nanmean(MHW_cum_ts_SA_MODEL, axis=(0, 1))
Cum_AL_ts = np.nanmean(MHW_cum_ts_AL_MODEL, axis=(0, 1))
Cum_BAL_ts = np.nanmean(MHW_cum_ts_BAL_MODEL, axis=(0, 1))
Cum_NA_ts = np.nanmean(MHW_cum_ts_NA_MODEL, axis=(0, 1))
#BMHWs
Bottom_Cum_CAN_ts = np.nanmean(BMHW_cum_ts_CAN_MODEL, axis=(0, 1))
Bottom_Cum_SA_ts = np.nanmean(BMHW_cum_ts_SA_MODEL, axis=(0, 1))
Bottom_Cum_AL_ts = np.nanmean(BMHW_cum_ts_AL_MODEL, axis=(0, 1))
Bottom_Cum_BAL_ts = np.nanmean(BMHW_cum_ts_BAL_MODEL, axis=(0, 1))
Bottom_Cum_NA_ts = np.nanmean(BMHW_cum_ts_NA_MODEL, axis=(0, 1))

#Replacing NaNs with 0s
Cum_CAN_ts = np.nan_to_num(Cum_CAN_ts, nan=0.0)
Cum_SA_ts = np.nan_to_num(Cum_SA_ts, nan=0.0)
Cum_AL_ts = np.nan_to_num(Cum_AL_ts, nan=0.0)
Cum_BAL_ts = np.nan_to_num(Cum_BAL_ts, nan=0.0)
Cum_NA_ts = np.nan_to_num(Cum_NA_ts, nan=0.0)


#Calculating MHW Fraction of Area (FoA)
threshold = 0.01
fraction_area_NA = calculate_fraction_area(MHW_cum_ts_NA_MODEL, threshold)
fraction_area_AL = calculate_fraction_area(MHW_cum_ts_AL_MODEL, threshold)
fraction_area_CAN = calculate_fraction_area(MHW_cum_ts_CAN_MODEL, threshold)
fraction_area_SA = calculate_fraction_area(MHW_cum_ts_SA_MODEL, threshold)
fraction_area_BAL = calculate_fraction_area(MHW_cum_ts_BAL_MODEL, threshold)

Bottom_fraction_area_NA = calculate_fraction_area(BMHW_cum_ts_NA_MODEL, threshold)
Bottom_fraction_area_AL = calculate_fraction_area(BMHW_cum_ts_AL_MODEL, threshold)
Bottom_fraction_area_CAN = calculate_fraction_area(BMHW_cum_ts_CAN_MODEL, threshold)
Bottom_fraction_area_SA = calculate_fraction_area(BMHW_cum_ts_SA_MODEL, threshold)
Bottom_fraction_area_BAL = calculate_fraction_area(BMHW_cum_ts_BAL_MODEL, threshold)

#Normalizing FoA
Max_fraction_area_NA = np.nanmax(fraction_area_NA, axis=0)
Max_fraction_area_AL = np.nanmax(fraction_area_AL, axis=0)
Max_fraction_area_CAN = np.nanmax(fraction_area_CAN, axis=0)
Max_fraction_area_SA = np.nanmax(fraction_area_SA, axis=0)
Max_fraction_area_BAL = np.nanmax(fraction_area_BAL, axis=0)

Bottom_Max_fraction_area_NA = np.nanmax(Bottom_fraction_area_NA, axis=0)
Bottom_Max_fraction_area_AL = np.nanmax(Bottom_fraction_area_AL, axis=0)
Bottom_Max_fraction_area_CAN = np.nanmax(Bottom_fraction_area_CAN, axis=0)
Bottom_Max_fraction_area_SA = np.nanmax(Bottom_fraction_area_SA, axis=0)
Bottom_Max_fraction_area_BAL = np.nanmax(Bottom_fraction_area_BAL, axis=0)

norm_fraction_area_NA = fraction_area_NA / Max_fraction_area_NA
norm_fraction_area_AL = fraction_area_AL / Max_fraction_area_AL
norm_fraction_area_CAN = fraction_area_CAN / Max_fraction_area_CAN
norm_fraction_area_SA = fraction_area_SA / Max_fraction_area_SA
norm_fraction_area_BAL = fraction_area_BAL / Max_fraction_area_BAL

Bottom_norm_fraction_area_NA = Bottom_fraction_area_NA / Bottom_Max_fraction_area_NA
Bottom_norm_fraction_area_AL = Bottom_fraction_area_AL / Bottom_Max_fraction_area_AL
Bottom_norm_fraction_area_CAN = Bottom_fraction_area_CAN / Bottom_Max_fraction_area_CAN
Bottom_norm_fraction_area_SA = Bottom_fraction_area_SA / Bottom_Max_fraction_area_SA
Bottom_norm_fraction_area_BAL = Bottom_fraction_area_BAL / Bottom_Max_fraction_area_BAL


#Ensure that the fraction of area is between 0 and 1
norm_fraction_area_NA = np.clip(norm_fraction_area_NA, 0, 1)
norm_fraction_area_AL = np.clip(norm_fraction_area_AL, 0, 1)
norm_fraction_area_CAN = np.clip(norm_fraction_area_CAN, 0, 1)
norm_fraction_area_SA = np.clip(norm_fraction_area_SA, 0, 1)
norm_fraction_area_BAL = np.clip(norm_fraction_area_BAL, 0, 1)



#Plotting the figure
fig, (axs1, axs2, axs3, axs4, axs5) = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})
norm = colors.Normalize(vmin=5, vmax=40)
norm_bottom = colors.Normalize(vmin=0, vmax=35)
cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.YlOrRd)
# cmap = plt.cm.ScalarMappable(norm=norm, cmap=cm.cm.thermal)
cmap_bottom = plt.cm.ScalarMappable(norm=norm_bottom, cmap=plt.cm.YlOrRd)


#North Atlantic
axs1.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding normalized bars
cs1 = axs1.bar(time, norm_fraction_area_NA, color=cmap.to_rgba(Cum_NA_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs1):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_NA[i]
    axs1.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css1 = axs1.scatter(time, Bottom_norm_fraction_area_NA, color=cmap_bottom.to_rgba(Bottom_Cum_NA_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css1.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_NA[i]
    axs1.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)

axs1.set_xlim(1992, 2023)
axs1.set_ylim(0, 1, 0.25)
axs1.set_yticks([0, 0.5, 1])
axs1.tick_params(length=5, direction='out', which='both', right=True)
axs1.yaxis.set_label_position('right')
axs1.text(0.02, 1.18, 'c', transform=axs1.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs1.set_title(r'North Atlantic (NA)', fontsize=14)


#Strait of Gibraltar and Alboran Sea
axs2.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs2 = axs2.bar(time, norm_fraction_area_AL, color=cmap.to_rgba(Cum_AL_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs2):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_AL[i]
    axs2.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css2= axs2.scatter(time, Bottom_norm_fraction_area_AL, color=cmap_bottom.to_rgba(Bottom_Cum_AL_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css2.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_AL[i]
    axs2.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)

axs2.set_xlim(1992, 2023)
axs2.set_ylim(0, 1, 0.25)
axs2.set_yticks([0, 0.5, 1])
axs2.tick_params(length=5, direction='out', which='both', right=True)
axs2.yaxis.set_label_position('right')
axs2.text(0.02, 1.18, 'f', transform=axs2.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs2.set_title(r'SoG and Alboran Sea (AL)', fontsize=14)


#Canary
axs3.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs3 = axs3.bar(time, norm_fraction_area_CAN, color=cmap.to_rgba(Cum_CAN_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs3):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_CAN[i]
    axs3.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css3 = axs3.scatter(time, Bottom_norm_fraction_area_CAN, color=cmap_bottom.to_rgba(Bottom_Cum_CAN_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css3.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_CAN[i]
    axs3.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)
    
axs3.set_xlim(1992, 2023)
axs3.set_ylim(0, 1, 0.25)
axs3.set_yticks([0, 0.5, 1])
axs3.tick_params(length=5, direction='out', which='both', right=True)
axs3.yaxis.set_label_position('right')
axs3.text(0.02, 1.18, 'i', transform=axs3.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs3.set_title(r'Canary (CAN)', fontsize=14)


#South Atlantic
axs4.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs4 = axs4.bar(time, norm_fraction_area_SA, color=cmap.to_rgba(Cum_SA_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs4):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_SA[i]
    axs4.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css4 = axs4.scatter(time, Bottom_norm_fraction_area_SA, color=cmap_bottom.to_rgba(Bottom_Cum_SA_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css4.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_SA[i]
    axs4.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)
    
axs4.set_xlim(1992, 2023)
axs4.set_ylim(0, 1, 0.25)
axs4.set_yticks([0, 0.5, 1])
axs4.tick_params(length=5, direction='out', which='both', right=True)
axs4.yaxis.set_label_position('right')
axs4.text(0.02, 1.18, 'l', transform=axs4.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs4.set_title(r'South Atlantic (SA)', fontsize=14)


#Levantine-Balearic
axs5.axhline(y=0.5, color='black', linestyle='--', alpha=0.2)

# Adding SMHWs bars
cs5 = axs5.bar(time, norm_fraction_area_BAL, color=cmap.to_rgba(Cum_BAL_ts))

# Adding normalized error lines in SMHW
for i, rect in enumerate(cs5):
    x = rect.get_x() + rect.get_width() / 2
    y = rect.get_height()
    error = normalized_error_Td_BAL[i]
    axs5.errorbar(x, y, yerr=error, color='black', alpha=0.5, capsize=5)

#BMHWs scatter
css5 = axs5.scatter(time, Bottom_norm_fraction_area_BAL, color=cmap_bottom.to_rgba(Bottom_Cum_BAL_ts), edgecolors='black', alpha=1, zorder=10)

# Extract the individual scatter points' coordinates
scatter_points = css5.get_offsets()

# Adding normalized error lines in BMHW
for i, point in enumerate(scatter_points):
    x, y = point  # Get x and y coordinates
    error = normalized_error_bottom_Td_BAL[i]
    axs5.errorbar(x, y, yerr=error, color='red', alpha=0.5, capsize=5)
    
axs5.set_xlim(1992, 2023)
axs5.set_ylim(0, 1, 0.25)
axs5.set_yticks([0, 0.5, 1])
axs5.tick_params(length=5, direction='out', which='both', right=True)
axs5.yaxis.set_label_position('right')
axs5.text(0.02, 1.18, 'o', transform=axs5.transAxes, ha='left', va='top', fontsize=14, weight='bold',
          bbox=dict(facecolor='none', edgecolor='none', pad=5, alpha=1), zorder=11)  # Add the title inside the axes box
axs5.set_title(r'Levantine-Balearic (BAL)', fontsize=14)

# Create common colorbar
# Create colorbar for SMHW
cbar_ax = fig.add_axes([0.15, 0.023, 0.62, 0.015])
cbar = plt.colorbar(cmap, cax=cbar_ax, extend='max', orientation='horizontal', format=ticker.FormatStrFormatter('%.0f'))
cbar.ax.tick_params(top=True, bottom=False, size=5, direction='out', which='both', labelsize=14)
cbar.ax.minorticks_off()
# cbar.set_ticks([10, 20, 30, 40])
cbar.ax.xaxis.set_ticks_position('top')
cbar.set_label(r'Averaged SMHW Cumulative Intensity [$^{\circ}C\ ·  days$]', fontsize=14, labelpad=-50)

# Create colorbar for BMHW
cbar_bottom_ax = fig.add_axes([0.15, 0.023, 0.62, 0.015])  # Adjust the position
cbar_bottom = plt.colorbar(cmap_bottom, cax=cbar_bottom_ax, extend='max', orientation='horizontal', format=ticker.FormatStrFormatter('%.0f'))
cbar_bottom.ax.tick_params(top=False, bottom=True, size=5, direction='out', which='both', labelsize=14)
cbar_bottom.ax.minorticks_off()
# cbar_bottom.set_ticks([2.5, 12.5, 20])
cbar_bottom.ax.xaxis.set_ticks_position('bottom')
cbar_bottom.set_label(r'Averaged BMHW Cumulative Intensity [$^{\circ}C\ ·  days$]', fontsize=14, labelpad=0)

fig.text(0.054, 0.5, 'Fraction of Area', va='center', rotation='vertical', fontsize=14)  # Agrega el título del eje y en el centro

plt.subplots_adjust(right=0.77, hspace=0.3)


outfile = r'...\Fig_5\Spatial_Extent_MHW_CumInt.png'
fig.savefig(outfile, dpi=600, bbox_inches='tight')
