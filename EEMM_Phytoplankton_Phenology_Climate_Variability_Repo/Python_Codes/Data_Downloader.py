# -*- coding: utf-8 -*-
"""
######################## Copernicus Data Downloader ###########################
"""

## Loading required python modules
import copernicusmarine
import os
import cdsapi

cd = r"D:\Paper_EEMM_Plankton_Phenology_Climate_Variability..."
os.chdir(cd)  
## Downloading subsets (you need to specify in console ur username and password)

# [CHL] (1×1 km)
copernicusmarine.subset(
  dataset_id="cmems_obs-oc_atl_bgc-plankton_my_l4-gapfree-multi-1km_P1D",
  variables=["CHL", "CHL_uncertainty"],
  minimum_longitude=-45.99479293823242,
  maximum_longitude=12.994793891906738,
  minimum_latitude=20.005207061767578,
  maximum_latitude=65.99478912353516,
  start_datetime="1998-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
)


# Peninsular SST (0.05° × 0.05°)
copernicusmarine.subset(
  dataset_id="cmems-IFREMER-ATL-SST-L4-REP-OBS_FULL_TIME_SERIE",
  variables=["analysed_sst", "analysis_error"],
  minimum_longitude=-20.975,
  maximum_longitude=12.975,
  minimum_latitude=8.925,
  maximum_latitude=61.975,
  start_datetime="1982-01-01T00:00:00",
  end_datetime="2022-12-31T00:00:00",
)
# Canary SST (0.05° × 0.05°)
copernicusmarine.subset(
  dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
  variables=["analysed_sst", "analysis_error"],
  minimum_longitude=-25.368175827861428,
  maximum_longitude=-6.712690889655139,
  minimum_latitude=22.465019252647934,
  maximum_latitude=34.74256062223669,
  start_datetime="1982-01-01T00:00:00",
  end_datetime="2022-05-31T00:00:00",
)


# MLD (0.083° × 0.083°)
copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
  variables=["mlotst"],
  minimum_longitude=-28.344214882389892,
  maximum_longitude=8.112061406903512,
  minimum_latitude=20.421766186544534,
  maximum_latitude=52.78601146377439,
  start_datetime="1993-01-01T00:00:00",
  end_datetime="2020-12-31T00:00:00",
  minimum_depth=0.49402499198913574,
  maximum_depth=0.49402499198913574,
)


# Wind Speed (5.5 km x 5.5 km)
dataset = "reanalysis-cerra-single-levels"
request = {
    "variable": ["10m_wind_speed"],
    "level_type": "surface_or_atmosphere",
    "data_type": ["reanalysis"],
    "product_type": "analysis",
    "year": [
        "1984", "1985", "1986",
        "1987", "1988", "1989",
        "1990", "1991", "1992",
        "1993", "1994", "1995",
        "1996", "1997", "1998",
        "1999", "2000", "2001",
        "2002", "2003", "2004",
        "2005", "2006", "2007",
        "2008", "2009", "2010",
        "2011", "2012", "2013",
        "2014", "2015", "2016",
        "2017", "2018", "2019",
        "2020", "2021", "2022",
        "2023"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": ["00:00"],
    "data_format": "netcdf"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
