#!/usr/bin/env python
# coding: utf-8

# # Calculate WBGT with CMIP6 data
# This notebook introduces how to use our code to calculate WBGT with 3hourly CMIP6 data during a 1-month period obtained from google cloud. Although it is not a big data problem, we still leverge Dask to do calculations lazily and parallelly which will be helpful for users to apply the code to large-size data.
# 
# The code was transferred (with a bit modifications) from Liljegren's WBGT code which is in C language (available from https://github.com/mdljts/wbgt/blob/master/src/wbgt.c). For details of the calculation procedure, please refer to Liljegren et al (2018).
# 
# To use our code, please consider cite: *Kong, Qinqin, and Matthew Huber. â€œExplicit Calculations of Wet Bulb Globe Temperature Compared with Approximations and Why It Matters for Labor Productivity.â€ Earthâ€™s Future, January 31, 2022. https://doi.org/10.1029/2021EF002334.*

# # 1. import packages

# ### 1.1 import general packages

# import general packages
import xarray as xr
import dask
import dask.array as da
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, vectorize
# import packages needed for obtaining google cloud data
import pandas as pd
import fsspec
import os
import sys
from pathlib import Path

# ### 1.2 import modules for WBGT calculation
# Our WBGT code was written in Cython (***.pyx* file** under ***src/***) which needs to be compiled first. We have compiled Cython source files and get shared object files (***.so* file**) which are put under ***src/***. The shared object file can be directly imported in Python as modules. Here we import two modules:
# 
# **(1)** ***coszenith***: the module for calculating $\cos\theta$ ($\theta$ denotes zenith angle). This is needed to project direct solar radiation from a flux through a horizontal plane (as stored in climate model output or reanalysis data) to a flux through a plane perpendicular to the incoming solar radiation (as required by Liljegren's WBGT model). Since solar radiation is generally stored as an accumulated or average quantity during an interval, here we calculate the average value of $\cos\theta$ during intervals of a certain length (e.g. hourly) defined by users.
# 
# From this module, we import two functions:
#   - ***cosza***: calculating the average $\cos\theta$ during each interval
#   - ***coszda***: calculating $\cos\theta$ during only the sunlit part of each interval; using the average $\cos\theta$ may lead to erroneously peaks of WBGT values around sunrise or sunset due to the small $\cos\theta$ values; these erroneously peaks can be removed by averaging $\cos\theta$ only during the sunlit period of each interval (see Kong and Huber (2021))
#   
# **(2)** ***WBGT***: the module for WBGT calculation from which we import:
#   - ***WBGT_Liljegren***: the original formulation in Liljegren's code. It only requires surface downward solar radiation as radiation input, and other radiation components (reflected solar radiation, and downward and upwelling long-wave radiation) are approximated.
#   - ***WBGT_GCM***: our modified formulation to directly use the full set of radiation components that are generally available from climate model output.


from coszenith import coszda, cosza
from WBGT import WBGT_Liljegren, WBGT_GCM, fdir


# # 2. Read in data 


ddir= '/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'
#read in different files
def open_ds(var,case,esm='cesm',model='cam',ens='001',stream='h0'):
    if esm=='cesm':
        if case[:4]=='hist':
            comp='BHIST'
            tseriesdir = ddir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-hist-i308.'+ens+'/' 
        else:
            comp='BSSP126'
            tseriesdir = ddir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ens+'/' 

        # define filename
        fn = var+'_b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ ens + '.' + model + '.' + stream + '.nc'
    elif esm=='mpiesm':
        ens='i'+ens[-1]

        if case=='hist':
            case='histctl'
            tspan='1980-2014'
        else:
            tspan='2015-2099'
        tseriesdir = ddir + esm+'/'+case+'/'
        # define filename
        if stream=='h0':
            fn = var+'_monthly_'+esm+'_'+case+'_'+ens+'_no-dynveg_'+tspan+ '.nc'
        else:
            fn = var+'_3hr_'+esm+'_'+case+'_'+ens+'_no-dynveg_'+tspan+ '.nc'
    elif esm=='ecearth':
        ens=ens[-1]
        if case=='hist':
            case='histctl'
        tseriesdir=ddir + esm+'/'+case+'/'
        fn= var+'*3hr_*_r'+ens+'*.nc'

    # check if variable timeseries exists and open variable as data array
    if not os.path.isfile(tseriesdir + fn) and esm !='ecearth':
        print(fn + ' does not exists in ')
        print(tseriesdir)
        return
    else: 
        
        if esm=='cesm':
            ds = xr.open_dataset(tseriesdir+fn)
            ds['time']=ds['time'].astype("datetime64[ns]")
        elif esm=='mpiesm':
           # ds = xr.open_dataset(savedir+fn)
            #print(ds.time.attrs['units'])
            ds = xr.open_dataset(tseriesdir+fn)
            backup_time=ds['time'].sel(time=slice('1985-01-01', None))
            ds['time']=pd.to_datetime(ds['time'], format='%Y%m%d') 
        elif esm=='ecearth':
            baseDir = Path(tseriesdir)
            ds= xr.open_mfdataset(baseDir.glob(fn))
        return ds

def open_da(var, case,esm='cesm', model='cam',ens='001',stream='h0'):
    ds = open_ds(var, case, esm,model,ens,stream)
    if var=='2t':
        var='var167'
    da = ds[var]
    return da


# read in all veraibles

esm=sys.argv[1]
case=sys.argv[2]
ens=sys.argv[3]

print(case+'_'+ens+'_'+esm)

if esm=='cesm':
	var_list=['TREFHT','RH2M','SWdown','WIND','PS']
else:
        var_list=['tas','hurs','rsds','sfcWind','ps']

tas=open_da(var_list[0],case,esm,'cam',ens,'h2').sel(time=slice('1985-01-01', None)).chunk({'time':8,'lat':8,'lon':8})
hurs=open_da(var_list[1],case,esm,'clm2',ens,'h6').sel(time=slice('1985-01-01', None)).chunk({'time':8,'lat':8,'lon':8})
rsds=open_da(var_list[2],case,esm,'clm2',ens,'h6').sel(time=slice('1985-01-01', None)).chunk({'time':8,'lat':8,'lon':8})
sfcWind=open_da(var_list[3],case,esm,'clm2',ens,'h6').sel(time=slice('1985-01-01', None)).chunk({'time':8,'lat':8,'lon':8})
ps=open_da(var_list[4],case,esm,'cam',ens,'h2').sel(time=slice('1985-01-01', None)).chunk({'time':8,'lat':8,'lon':8})

if esm=='cesm':   ##slight inconsistencies in coords land model variables and atm -> just use land model here
      tas['lat']=rsds['lat']
      tas['lon']=rsds['lon']
      ps['lat']=rsds['lat']
      ps['lon']=rsds['lon']

if esm=='mpiesm':
      hurs=hurs*100
if esm=='ecearth':
      rsds['time']=tas['time']

# # 3. Data preprocessing
# 
# Before calling the function to calculate WBGT, we need to do some preprocessing on the data.

# ### 3.1 Interpolate surface pressure
# First, we notice that there is an offset between the time point of instantaneous nonradiation fields (e.g. temperature, humidity and wind) and the center point of the interval across which radiation fluxes are averaged. In this case, temperature, humidity and wind fields are located at 00:00:00, 03:00:00, 06:00:00..., whereas radiation fields are centered on 01:30:00, 04:30:00, 07:30:00 ... Hence, we need to interpolate radiation fields to the same time points as other fields.
# function to interpolate radiation fields
def interp(data):
    date=xr.DataArray(tas.time.values,dims=('time'),coords={'time':tas.time})
    # notice that the first time step "2001-07-01T00:00:00" will be interpolated as NaN
    result=data.interp(time=date,method='linear')
    return result.chunk({'time':8})
#if esm=='cesm':
#	ps_interp=interp(ps)
#	ps_interp['lat']=rsds['lat']
#	ps_interp['lon']=rsds['lon']
#else:
#	ps_interp=ps
ps_interp=ps



# ### 3.2 calculate $\cos\theta$
# We calculate $\cos\theta$ averaged both during each interval and during only the sunlit period of each interval. Both of them are needed for calculating WBGT. For the details of the calculation procedure of $\cos\theta$ please refer to Hogan and Hirahara (2016), and Di Napoli et al (2020)



# create meshgrid of latitude and longitude, and we will calculate cosine zenith angle on these grids
lon,lat=np.meshgrid(tas.lon.chunk({'lon':8}),tas.lat.chunk({'lat':8}))
lat=lat*np.pi/180
lon=lon*np.pi/180


# calculate $\cos\theta$ averaged during each interval.
# 
# the function ***cosza*** asks for four inputs: 
# - a time series (UTC time) indicating the center points of intervals for which the $\cos\theta$ will be calculated
# - the latitude and longitude arrays across which $\cos\theta$ will be calculated
# - the length of each interval (e.g. 3 for 3hourly)


# specifiy the time seris for which we want to calculate cosine zenith angle 
if esm=='cesm':
    date2 = tas['time'].astype("datetime64[ns]").chunk({'time':8})
    date=xr.DataArray(date2.values,dims=('time'),coords={'time':date2}).chunk({'time':8})
else:
    date=xr.DataArray(tas.time.values,dims=('time'),coords={'time':tas.time}).chunk({'time':8})
    
# use dask.array map_blocks to calculate cosine zenith angle lazily and parallelly
cza=da.map_blocks(cosza,date.data,lat,lon,3,chunks=(8,lat.shape[0],lat.shape[1]),new_axis=[1,2])
if esm=='cesm':
    cza=cza[:-7,:,:]   ##cesm artificially creates an extra day in 2015 which we remove here
# transfer to xarray DataArray
cza=xr.DataArray(cza,dims=tas.dims, coords=tas.coords) 


# calculate $\cos\theta$ averaged during only the sunlit part of each interval.
# 
# the function ***coszda*** asks for four inputs: 
# - a time series (UTC time) indicating the center points of intervals for which the $\cos\theta$ will be calculated
# - the latitude and longitude arrays across which $\cos\theta$ will be calculated
# - the length of each interval (e.g. 3 for 3hourly)


czda=da.map_blocks(coszda,date.data,lat,lon,3,chunks=(8,lat.shape[0],lat.shape[1]),new_axis=[1,2])
if esm=='cesm':
    czda=czda[:-7,:,:]   ##cesm artificially creates an extra day in 2015 which we remove here

# transfer to xarray DataArray
czda=xr.DataArray(czda,dims=tas.dims, coords=tas.coords)
# here we change zero (czda=0 when the sun is below horizon) to an arbitrary negative value (here we choose -0.5) to avoid division by zero 
czda=xr.where(czda<=0,-0.5,czda)


# ### 3.3 Calculate relative humidity
# Here we calculate relative humidity from specific humidity since relative humidity is one of the required inputs for WBGT function.


# calculate saturated vapor pressure
@vectorize
def esat(tas,ps):
    # tas: temperature (K)
    # ps: surface pressure (Pa)
    # return saturation vapor pressure (Pa)
    if tas>273.15:
        es = 611.21 * np.exp(17.502 * (tas - 273.15) *((tas - 32.18)**(-1)))
        es = (1.0007 + (3.46*10**(-6) * ps/100)) * es
    else:
        es = 611.15 * np.exp(22.452 * (tas - 273.15) * ((tas - 0.6)**(-1)))
        es=(1.0003 + (4.18*10**(-6) * ps/100)) * es
    return es
# calculate vapor pressure
def vaporpres(huss, ps):
    # huss: specific humidity (kg/kg)
    # ps: surface pressure (Pa)
    # return vapor pressure (Pa)
    r=huss*((1-huss)**(-1))
    return ps*r*((0.622+r)**(-1))
# calculate relative humidity from specific humidity
def huss2rh(tas,huss,ps):
    # tas: temperature (K)
    # huss: specific humidity (kg/kg)
    # ps: surface pressure (Pa)
    # return relative humidity (%)
    return vaporpres(huss, ps)*(esat(tas,ps)**(-1))*100


# calculate relative humidity
#hurs=xr.apply_ufunc(huss2rh,tas,huss,ps,dask="parallelized",output_dtypes=[float])
# set the maximum value as 100%
#hurs=xr.where(hurs>100,100,hurs)


# ### 3.5 Calculate the ratio of direct solar radiation
# We need to know the ratio of direct solar radiation in order to get the radiation heat gain right for WBGT calculation, since the treatment of direct and diffuse solar radiation are different in the code. 
# 
# In this case, we have ```rsdsdiff``` which is the diffuse solar radiation, so we can directly calculate the ratio of direct solar radiation. 
# 
# When we don't have such a field, we need to call the **```fdir```** function (also can be imported from the ```WBGT``` module) to calculate the ratio. Please refer to the ```WBGT.pyx``` file for how to call **```fdir```** function.


#f=(rsdsinterp-rsdsdiffinterp)/rsdsinterp 
f=xr.apply_ufunc(fdir,cza,czda,rsds,date,dask="parallelized",output_dtypes=[float])
    # cza: temporal average cosine zenith angle during each interval
    # czda: temporal average cosine zenith angle during only the sunlit part of each interval
    # rsds: surface downward solar radiation (w/m2)
    # date: date and time series that you want to calculate
    # return the ratio of direct solar radiation 

# the treatments below aim to avoid unrealistically high or low values of f which are also included in Liljegren's code.
f=xr.where(cza<=np.cos(89.5/180*np.pi),0,f) 
f=xr.where(f>0.9,0.9,f)
f=xr.where(f<0,0,f)
f=xr.where(rsds<=0,0,f)



# # 4. Calculate WBGT

# ### 4.1 Liljegren's original formulation
# In Liljegren's original formulation, downward solar radiation is the only required radiation input, other radiation components are approximated internally.
# 
# The meaning of each argument:
# - ```tas```: air temperature (K)
# -  ```hurs```: relative humidity (%)
# -  ```sfcwind```: 2 meter wind speed (m/s)
# -  ```ps```: surface pressure (Pa)
# -  ```rsdsinterp```: surface downward solar radiation (w/m2)
# -  ```f```: the ratio of direct solar radiation 
# - ```czda```: average cosine zenith angle during only the sunlit period of each interval
# 
# -  the ```False``` argument at the end tell the function that our wind speed is not at 2meter height which will make  the function to treat wind speed as 10 meter height and transfer it to 2 meter. The function currently only support 2 meter and 10meter wind. For wind speed at other heights, users need to change the source code slightly.


wbgt_liljegren=xr.apply_ufunc(WBGT_Liljegren,tas,hurs,ps_interp,sfcWind,rsds,f,czda,False,dask="parallelized",output_dtypes=[float])


#wbgt_liljegren.to_netcdf(ddir+'wbgt_liljegren_'+case+'_'+esm+'_'+ens+'.nc')
if esm !='mpiesm':
    ds = xr.Dataset(
        data_vars=dict(
           wbgt=(["time","lat","lon"], wbgt_liljegren.values),
        ),
         coords=dict(
           lon=(["lon"], tas.lon.values),
           lat=(["lat"], tas.lat.values),
           time=(["time"], tas.time.values),
           #reference_time=reference_time,
        ), 
    )
else:
    ds = xr.Dataset(
        data_vars=dict(
           wbgt=(["time","lat","lon"], wbgt_liljegren.values),
        ),
         coords=dict(
           lon=(["lon"], tas.lon.values),
           lat=(["lat"], tas.lat.values),
           time=(["time"], backup_time.values),
           #reference_time=reference_time,
        ),
    )

ds.to_netcdf('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/wbgt_liljegren_corr_'+case+'_'+esm+'_'+ens+'.nc', encoding={'time': {'dtype': 'i4'}})


# # References
# Liljegren, J. C., Carhart, R. A., Lawday, P., Tschopp, S. & Sharp, R. Modeling the Wet Bulb Globe Temperature Using Standard Meteorological Measurements. Journal of Occupational and Environmental Hygiene 5, 645â€“655 (2008).
# 
# Di Napoli, C., Hogan, R. J. & Pappenberger, F. Mean radiant temperature from global-scale numerical weather prediction models. Int J Biometeorol 64, 1233â€“1245 (2020).
# 
# Hogan, R. J. & Hirahara, S. Effect of solar zenith angle specification in models on mean shortwave fluxes and stratospheric temperatures. Geophys. Res. Lett. 43, 482â€“488 (2016).
# 
