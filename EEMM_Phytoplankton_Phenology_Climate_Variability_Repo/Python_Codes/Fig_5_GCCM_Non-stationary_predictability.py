# -*- coding: utf-8 -*-
"""

###### Fig. 5. Causal Inference (GCCM) - Non-stationary predictability of 
       phytoplankton dynamics (NA) ########

"""

#Loading required Python modules
import numpy as np
import xarray as xr
import pandas as pd

import pycwt as wavelet

import matplotlib.pyplot as plt

import pyEDM

import seaborn as sns

from joblib import Parallel, delayed



## Fig. 5a-c - CHL Wavelet analysis (North Atlantic - NA) ##

#Preparing Dataset
# ds = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_NA.nc')
# ds_smoothed = ds.rolling(time=21, center=True, min_periods=1).mean()

# dat = ds_smoothed.CHL.mean(dim=('lat', 'lon'), skipna=True)
# data = dat.values.squeeze()

# np.save(r'E:\...\Wavelet_Data/NA.npy', data)

# Load previously-proccessed data 
ds = xr.open_dataset(r'E:\...\CHL_L4_1Km_Data/CHL_L4_1km_NA.nc')
data = np.load(r'E:\...\Wavelet_Data/NA.npy')

# Extract time information from the dataset
time = ds.CHL['time'].values
dt = np.timedelta64(int((time[1] - time[0]) / np.timedelta64(1, 's')), 's').astype(float) / (365.25 * 24 * 3600)

# Other parameters
label = '[Chl-a]'
units = r' mg$\cdot$m$^{-3}$'
t0 = 1998.0 # Dataset starts in 1998
years = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
# Preprocessing
N = data.size
t = np.arange(0, N) * dt + t0

p = np.polyfit(t - t0, data.flatten(), 1)
# dat_notrend = data - np.polyval(p, t - t0)
# std = dat_notrend.std()  # Standard deviation
std = np.nanstd(data)  # Standard deviation
var = std ** 2  # Variance
# dat_norm = dat_notrend / std  # Normalized dataset

# Wavelet parameters
mother = wavelet.Morlet(6)
s0 = 0.05  # Starting scale, in this case 2 * 0.0027 years = 312 months
dj = 0.03  # Twelve sub-octaves per octaves
J = 7 / dj  # Seven powers of two with dj sub-octaves
alpha, _, _ = wavelet.ar1(data.flatten())  # Lag-1 autocorrelation for red noise

# Continuous wavelet transform
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(data.flatten(), dt, dj, s0, J, mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

# Power and significance calculations
power = (np.abs(wave)) ** 2
fft_power = np.abs(fft) ** 2
period = 1 / freqs

power /= scales[:, None]

signif, fft_theor = wavelet.significance(1.75, dt, scales, 0, alpha, significance_level=0.95, wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95

glbl_power = power.mean(axis=1)
dof = N - scales  # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(0.035, dt, scales, 1, alpha, significance_level=0.95, dof=dof, wavelet=mother)

sel = np.where((period >= 0.0625) & (period < 2))[0]

Cdelta = mother.cdelta
scale_avg = (scales * np.ones((N, 1))).transpose()
scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
if len(sel) > 0:
    scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha, significance_level=0.95, dof=[scales[sel[0]], scales[sel[-1]]], wavelet=mother)
else:
    # Handle the case when sel is empty, for example, by setting scale_avg_signif to a default value
    scale_avg_signif = 0.0  # Replace with an appropriate default value

# Fig. 5a, the normalized wavelet power spectrum and significance
# level contour lines and cone of influence hatched area. Note that period
# scale is logarithmic.
fig, axs = plt.subplots(2, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [2, 1]})
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32] 
axs[0,0].contourf(t, np.log2(period), np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.Spectral_r)
extent = [t.min(), t.max(), min(period), max(period)]
sig95_levels = [np.percentile(power, 95.5)]
axs[0,0].contour(t, np.log2(period), power, levels=sig95_levels, colors='k', linewidths=1.5, extent=extent, linestyles='--')
axs[0,0].fill(
    np.concatenate([t, t[::-1]]),  
    np.concatenate([np.log2(coi), np.log2(period[-1:]) * np.ones_like(t[::-1])]), 
    'k', alpha=0.4, hatch='x')

axs[0,0].set_title('{} Wavelet Power Spectrum'.format(label))
axs[0,0].set_ylabel('Period [years]', fontsize=16)

axs[0,0].set_ylim([np.log2(2.25), np.log2(0.05)])
Yticks = 2 ** np.arange(np.floor(np.log2(0.05)), np.ceil(np.log2(2.25)))
Yticks = Yticks[1:7]
axs[0,0].set_yticks(np.log2(Yticks))
axs[0,0].set_yticklabels(Yticks)
axs[0,0].set_xticks(np.round((t[::1460])))
axs[0,0].set_xticklabels(years)
axs[0,0].set_aspect('auto')
 
# Fig. 5b, the global wavelet and Fourier power spectra and theoretical
# noise spectra. Note that period scale is logarithmic.
axs[0,1].plot(glbl_signif, np.log2(period), 'k--')
axs[0,1].plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
axs[0,1].plot(var * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc', linewidth=1.)
axs[0,1].plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
axs[0,1].set_title('General Wavelet Spectrum')
axs[0,1].set_xlabel(r'$\mathrm{Power\ }[(\mathrm{mg} \cdot \mathrm{m}^{-3})^2]$', fontsize=16)
axs[0,1].set_xlim([0, 2.5])
axs[0,1].set_ylim(np.log2([period.min(), period.max()]))
axs[0,1].set_yticks(np.log2(Yticks))
axs[0,1].set_yticklabels(Yticks)
axs[0,1].set_ylim([np.log2(2.25), np.log2(0.05)])
plt.setp(axs[0,1].get_yticklabels(), visible=False)

# Fig. 5c, the scale-averaged Wavelet Spectrum.
axs[1,0].axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
axs[1,0].plot(t, scale_avg, 'k-', linewidth=1.5)
axs[1,0].set_title('Scale-averaged power', fontsize=18)
# axs[1,0].set_xlabel('Time [year]')
axs[1,0].set_ylabel(r'$\mathrm{Variance}$' '\n' r'$\mathrm{[mg \cdot m^{-3}]}$', fontsize=16)
axs[1,0].set_xlim([t.min(), t.max()])
axs[1,0].set_ylim([0, 0.075])
axs[1,0].set_xticks(np.round((t[::1460])))
axs[1,0].set_xticklabels(years)
axs[1,0].set_yticks([0.01, 0.04, 0.075])
axs[1,0].set_yticklabels(['0.01', '0.04', '0.08'])

axs[1, 1].axis('off')

for ax in axs.flat:
    ax.tick_params(axis='both', which='both', labelsize=16)  
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)  

plt.tight_layout()
plt.show()


fig.savefig(r'E:\...\Figures\Fig_5/Fig_5a-c.png', dpi=1200, bbox_inches='tight')



## Fig. 5d, f - Causal Inference through GCCM (Gao et al., 2023; Sugihara et al., 2012) ##

# Loading MHW Cum Intensity (from Fernández-Barba et al. (2024)), Windiness, BMaxChl-a, and SCR datasets

MHWCumInt = np.load(r'E:\...\Annual_Data\MHWCumInt_Interp_Data/MHWCumInt_NA.npy')
MHWCumInt = np.where(np.isnan(MHWCumInt), 0, MHWCumInt)
MHWCumInt = MHWCumInt[::10, ::10, :]
Windiness = np.load(r'E:\...\Annual_Data\Windiness/Windiness_NA.npy')
Windiness = Windiness[::10, ::10, :]

BMaxChla = np.load(r'E:\...\Phenology_Metrics\NA/bloom_max_NA.npy')
# BMaxChla = BMaxChla[::10,::10, 0:25]
BMaxChla = BMaxChla[::10,::10, :]

SCR = np.load(r'E:\...\Phenology_Metrics\NA/SCR_NA.npy')
# SCR = SCR[::10,::10, 0:25]
SCR = SCR[::10,::10, :]

# Embedding and lag parameters
E = 6
tau = 1

# Variables to store results
rho_X_to_Y_values = []
rho_Y_to_X_values = []
lib_sizes = []

# Iterate over each year
# for year in range(25):     #MHWCumInt
for year in range(26):   #Windiness
    print(f"Processing year: {1998 + year}")
    
    # Extract data for the current year
    X = Windiness[:, :, year].flatten()
    Y = BMaxChla[:, :, year].flatten()

    # Create a pandas DataFrame
    df = pd.DataFrame({'X': X, 'Y': Y})
    
    # Define library size
    N = len(X)
    lib_sizes_for_year = np.linspace(10, N, 10, dtype=int)
    
    # Store the library sizes used
    if year == 0:
        lib_sizes = lib_sizes_for_year
    
    # Perform CCM for each library size
    rho_X_to_Y_year = []
    rho_Y_to_X_year = []
    
    for lib_size in lib_sizes_for_year:
        # CCM for X predicting Y
        ccm_results_X_to_Y = pyEDM.CCM(dataFrame=df, 
                                       E=E, tau=tau, 
                                       columns="X", target="Y", 
                                       libSizes=[lib_size], sample=1)
        
        # Access the 'X:Y' column that contains the cross-map skill for X predicting Y
        if 'X:Y' in ccm_results_X_to_Y.columns:
            rho_X_to_Y = ccm_results_X_to_Y['X:Y'][0]  # Cross-map skill (rho) for this library size
            rho_X_to_Y_year.append(rho_X_to_Y)
        else:
            print(f"'X:Y' not found in the result for lib_size={lib_size}")
            rho_X_to_Y_year.append(np.nan)  # Handle the case where 'X:Y' is not found

        # CCM for Y predicting X
        ccm_results_Y_to_X = pyEDM.CCM(dataFrame=df, 
                                       E=E, tau=tau, 
                                       columns="Y", target="X", 
                                       libSizes=[lib_size], sample=1)
        
        # Access the 'Y:X' column that contains the cross-map skill for Y predicting X
        if 'Y:X' in ccm_results_Y_to_X.columns:
            rho_Y_to_X = ccm_results_Y_to_X['Y:X'][0]  # Cross-map skill (rho) for this library size
            rho_Y_to_X_year.append(rho_Y_to_X)
        else:
            print(f"'Y:X' not found in the result for lib_size={lib_size}")
            rho_Y_to_X_year.append(np.nan)  # Handle the case where 'Y:X' is not found
    
    rho_X_to_Y_values.append(rho_X_to_Y_year)
    rho_Y_to_X_values.append(rho_Y_to_X_year)

# Convert results into an array for easier manipulation
rho_X_to_Y_values = np.array(rho_X_to_Y_values)
rho_Y_to_X_values = np.array(rho_Y_to_X_values)

np.save(r'E:\...\GCCM_outputs\BMaxChla_xmap_Windiness/BMaxChla_xmap_Windiness_NA.npy', rho_X_to_Y_values)
np.save(r'E:\...\GCCM_outputs\BMaxChla_xmap_Windiness/Windiness_xmap_BMaxChla_NA.npy', rho_Y_to_X_values)
np.save(r'E:\...\GCCM_outputs\libsizes/libsizes_NA.npy', lib_sizes)


# Fig. 5d (BMaxChl-a/SCR xmap MHW Cum Intensity)
libsizes_NA = np.load(r'E:\...\GCCM_outputs\libsizes/libsizes_NA.npy')
BMaxChla_xmap_MHWCumInt_NA = np.load(r'E:\...\GCCM_outputs\BMaxChla_xmap_MHWCumInt/BMaxChla_xmap_MHWCumInt_NA.npy')
MHWCumInt_xmap_BMaxChla_NA = np.load(r'E:\...\GCCM_outputs\BMaxChla_xmap_MHWCumInt/MHWCumInt_xmap_BMaxChla_NA.npy')
SCR_xmap_MHWCumInt_NA = np.load(r'E:\...\GCCM_outputs\SCR_xmap_MHWCumInt/SCR_xmap_MHWCumInt_NA.npy')

years = np.arange(1998, 2023)

data = []

for i, year in enumerate(years):
    for value in BMaxChla_xmap_MHWCumInt_NA[i, :]:
        data.append([year, value, 'BMaxChla_xmap_MHWCumInt_NA'])
    for value in MHWCumInt_xmap_BMaxChla_NA[i, :]:
        data.append([year, value, 'MHWCumInt_xmap_BMaxChla_NA'])

df = pd.DataFrame(data, columns=['Year', 'Value', 'Dataset'])

df['Dataset'] = df['Dataset'].replace({
    'BMaxChla_xmap_MHWCumInt_NA': 'BMaxChl-a xmap MHW Cum Intensity',
    'MHWCumInt_xmap_BMaxChla_NA': 'MHW Cum Intensity xmap BMaxChl-a'
})


df['Year_str'] = df['Year'].astype(str)

df_bmaxchla_xmap = df[df['Dataset'] == 'BMaxChl-a xmap MHW Cum Intensity']
df_mhwcumint_xmap = df[df['Dataset'] == 'MHW Cum Intensity xmap BMaxChl-a']

fig, ax = plt.subplots(figsize=(10, 5))

sns.violinplot(x='Year_str', y='Value', hue='Dataset', data=df_bmaxchla_xmap, width=2, 
               palette={'BMaxChl-a xmap MHW Cum Intensity': 'lightgreen'}, alpha=0.7, ax=ax, dodge=False, split=False)

sns.violinplot(x='Year_str', y='Value', hue='Dataset', data=df_mhwcumint_xmap, width=4, 
               palette={'MHW Cum Intensity xmap BMaxChl-a': 'lightpink'}, alpha=0.7, ax=ax, dodge=False, split=False)

mean_values = [np.nanmean(SCR_xmap_MHWCumInt_NA[i, :]) for i in range(SCR_xmap_MHWCumInt_NA.shape[0])]
ax.plot(df['Year_str'].unique(), mean_values, 'o', color='#FFFACD', label='SCR xmap MHW Cum Intensity', markersize=10, markeredgecolor='black', markeredgewidth=0.5)

plt.title('Geographical CCM: North Atlantic (NA)', fontsize=16)
plt.ylabel('Cross Map Skill [ρ]', fontsize=16)
plt.xlabel('')
plt.ylim([0, 1.2])

xticks = np.arange(0, len(years), 1)
ax.set_xticks(xticks)
xticklabels = [str(year) if (year - 1998) % 4 == 0 else '' for year in np.arange(1998, 1998 + len(years))]
ax.set_xticklabels(xticklabels, fontsize=16)
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)   
plt.yticks(fontsize=16)
plt.legend(fontsize=16, frameon=False, loc='upper left')
plt.tight_layout()
plt.show()


outfile = r'E:\...\Figures\Fig_5\Fig_5d.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



# Fig. 5f (BMaxChl-a/SCR xmap Windiness)
libsizes_NA = np.load(r'E:\...\GCCM_outputs\libsizes/libsizes_NA.npy')
BMaxChla_xmap_Windiness_NA = np.load(r'E:\...\GCCM_outputs\BMaxChla_xmap_Windiness/BMaxChla_xmap_Windiness_NA.npy')
Windiness_xmap_BMaxChla_NA = np.load(r'E:\...\GCCM_outputs\BMaxChla_xmap_Windiness/Windiness_xmap_BMaxChla_NA.npy')
SCR_xmap_Windiness_NA = np.load(r'E:\...\GCCM_outputs\SCR_xmap_Windiness/SCR_xmap_Windiness_NA.npy')

years = np.arange(1998, 2024)

data = []

for i, year in enumerate(years):
    for value in BMaxChla_xmap_Windiness_NA[i, :]:
        data.append([year, value, 'BMaxChla_xmap_Windiness_NA'])
    for value in Windiness_xmap_BMaxChla_NA[i, :]:
        data.append([year, value, 'Windiness_xmap_BMaxChla_NA'])

df = pd.DataFrame(data, columns=['Year', 'Value', 'Dataset'])

df['Dataset'] = df['Dataset'].replace({
    'BMaxChla_xmap_Windiness_NA': 'BMaxChl-a xmap Windiness',
    'Windiness_xmap_BMaxChla_NA': 'Windiness xmap BMaxChl-a'
})


df['Year_str'] = df['Year'].astype(str)

df_bmaxchla_xmap = df[df['Dataset'] == 'BMaxChl-a xmap Windiness']
df_Windiness_xmap = df[df['Dataset'] == 'Windiness xmap BMaxChl-a']

fig, ax = plt.subplots(figsize=(10, 5))

sns.violinplot(x='Year_str', y='Value', hue='Dataset', data=df_bmaxchla_xmap, width=5, 
               palette={'BMaxChl-a xmap Windiness': 'lightblue'}, alpha=0.7, ax=ax, dodge=False, split=False)

sns.violinplot(x='Year_str', y='Value', hue='Dataset', data=df_Windiness_xmap, width=4, 
               palette={'Windiness xmap BMaxChl-a': 'lightpink'}, alpha=0.7, ax=ax, dodge=False, split=False)

mean_values = [np.nanmean(SCR_xmap_Windiness_NA[i, :]) for i in range(SCR_xmap_Windiness_NA.shape[0])]
ax.plot(df['Year_str'].unique(), mean_values, 'o', color='#FFFACD', label='SCR xmap Windiness', markersize=10, markeredgecolor='black', markeredgewidth=0.5)

plt.title('Geographical CCM: North Atlantic (NA)', fontsize=16)
plt.ylabel('Cross Map Skill [ρ]', fontsize=16)
plt.xlabel('')
plt.ylim([0, 1])

xticks = np.arange(0, len(years), 1)
ax.set_xticks(xticks)
xticklabels = [str(year) if (year - 1998) % 4 == 0 else '' for year in np.arange(1998, 1998 + len(years))]
ax.set_xticklabels(xticklabels, fontsize=16)
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)   
plt.yticks(fontsize=16)
plt.legend(fontsize=16, frameon=False, loc='upper left')
plt.tight_layout()
plt.show()


outfile = r'E:\...\Figures\Fig_5\Fig_5f.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 5e, g - Causal Inference through CCM (Sugihara et al., 2012) ##
# Loading monthly MHW Cum Intensity (from Fernández-Barba et al. (2024)), Windiness, and BMaxChl-a datasets
Monthly_MHWCumInt = np.load(r'E:\...\Monthly_Data\Monthly_MHWCumInt_Interp_Data/Monthly_MHWCumInt_NA.npy')
Monthly_MHWCumInt = np.where(np.isnan(Monthly_MHWCumInt), 0, Monthly_MHWCumInt)

ds_Windiness = xr.open_dataset(r'E:\...\Monthly_Data\Monthly_Windiness/Monthly_Windiness_NA.nc')
ds_Windiness = ds_Windiness.transpose('longitude', 'latitude', 'time')
Windiness = ds_Windiness.wind_speed 
Windiness = Windiness.isel(time=slice(0, 311))

ds_Max_CHL = xr.open_dataset(r'E:\...\Monthly_Data\Monthly_MaxCHL/Monthly_MaxCHL_NA.nc')
ds_Max_CHL = ds_Max_CHL.transpose('lon', 'lat', 'time')
Max_CHL = ds_Max_CHL.CHL
Max_CHL = Max_CHL.sortby('lat')


embedding_values = range(1, 13) 
lag_values = range(1, 13)

def test_edm_sensitivity(windiness_series, maxchla_series, embedding_values, lag_values):
    results = []

    def process_parameters(E, tau):
        data = {
            'Time': np.arange(len(maxchla_series)),
            'Windiness': windiness_series.values,
            'Max_CHL': maxchla_series.values
        }
        df = pd.DataFrame(data).dropna()

        available_samples = len(df) - (E-1)*tau
        if available_samples <= 0:
            rho = np.nan
        else:
            sample_size = int(len(df))
            max_lib_size = available_samples
            
            lib_sizes_adjusted = False
            while not lib_sizes_adjusted:
                libsizes = np.array([max_lib_size], dtype=np.int32)
                try:
                    result = pyEDM.CCM(
                        dataFrame=df,
                        E=E,
                        tau=tau,
                        columns="Windiness",
                        target="Max_CHL",
                        libSizes=libsizes,
                        sample=sample_size
                    )

                    if not result.empty:
                        rho = result['Windiness:Max_CHL'].max()
                    else:
                        rho = np.nan
                    lib_sizes_adjusted = True
                except RuntimeError as e:
                    print(f"Adjusting for E={E}, tau={tau}, libSize={max_lib_size} due to error: {e}")
                    max_lib_size -= 1  
                    if max_lib_size < 1:
                        rho = np.nan
                        lib_sizes_adjusted = True

        return (E, tau, rho)

    results = Parallel(n_jobs=-1)(delayed(process_parameters)(E, tau) for E in embedding_values for tau in lag_values)

    return zip(*results)


target_lat = 42.4
target_lon = -9

closest_lat_idx = np.abs(Max_CHL.lat - target_lat).argmin()
closest_lon_idx = np.abs(Max_CHL.lon - target_lon).argmin()

windiness_series = Windiness.isel(latitude=closest_lat_idx, longitude=closest_lon_idx)
maxchla_series = Max_CHL.isel(lat=closest_lat_idx, lon=closest_lon_idx)

E_results, tau_results, rho_results = test_edm_sensitivity(windiness_series, maxchla_series, embedding_values, lag_values)


## Fig. 5e - Monthly BMaxChl-a xmap MHW Cum Intensity
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(12, 12), facecolor='white')
ax = fig.add_subplot(111, projection='3d', position=[0.1, 0.1, 0.65, 0.8])

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

tri_surf = ax.plot_trisurf(E_results, tau_results, rho_results, cmap=plt.cm.RdYlGn, alpha=0.8, vmin=0, vmax=0.8)

ax.set_xlabel('Embedding Dimension [E]', fontsize=22, labelpad=12)
ax.set_ylabel('Time lag in months [τ]', fontsize=22, labelpad=12)
ax.set_xlim(12, 1)  
ax.set_ylim(1, 12)
ax.set_zlim(0, 0.8)
ax.set_xticks([2, 4, 6, 8, 10, 12])
ax.set_yticks([2, 4, 6, 8, 10, 12])
ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8])

ax.view_init(elev=20, azim=51)

cbar = fig.colorbar(tri_surf, orientation='horizontal', shrink=0.5, aspect=20, pad=0.03)
cbar.set_label('ρ(BMaxChl-a xmap MHW Cum Intensity)', labelpad=-71, fontsize=22)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8])

ax.text2D(-0.055, 0.5, "Cross Map Skill [ρ]", transform=ax.transAxes, fontsize=22, va='center', ha='center', rotation=92.5)
plt.show()


outfile = r'E:\...\Figures\Fig_5\Fig_5e.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')



## Fig. 5g - Monthly BMaxChl-a xmap Windiness
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(12, 12), facecolor='white')
ax = fig.add_subplot(111, projection='3d', position=[0.1, 0.1, 0.65, 0.8])

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

tri_surf = ax.plot_trisurf(E_results, tau_results, rho_results, cmap=plt.cm.RdYlBu, alpha=0.8, vmin=0, vmax=0.6)

ax.set_xlabel('Embedding Dimension [E]', fontsize=22, labelpad=12)
ax.set_ylabel('Time lag in months [τ]', fontsize=22, labelpad=12)
ax.set_xlim(12, 1)  
ax.set_ylim(1, 12)
ax.set_zlim(0, 0.6)
ax.set_xticks([2, 4, 6, 8, 10, 12])
ax.set_yticks([2, 4, 6, 8, 10, 12])
ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8])

ax.view_init(elev=20, azim=51)

cbar = fig.colorbar(tri_surf, orientation='horizontal', shrink=0.5, aspect=20, pad=0.03)
cbar.set_label('ρ(BMaxChl-a xmap Windiness)', labelpad=-71, fontsize=22)
cbar.set_ticks([0, 0.2, 0.4, 0.6])

ax.text2D(-0.055, 0.5, "Cross Map Skill [ρ]", transform=ax.transAxes, fontsize=22, va='center', ha='center', rotation=92.5)
plt.show()


outfile = r'E:\...\Figures\Fig_5\Fig_5g.png'
fig.savefig(outfile, dpi=1200, bbox_inches='tight')

