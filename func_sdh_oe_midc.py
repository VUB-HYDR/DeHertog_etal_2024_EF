
### Same as func_sdh but for cmorized data (here mpiesm and ecearth)


from __future__ import print_function
import sys
import os
from getpass import getuser
import string
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import netCDF4 as netcdf4
import xarray as xr
import pandas
#import regionmask
#import cartopy.crs as ccrs
from IPython.display import display, Math, Latex
import warnings
from datetime import datetime
from datetime import timedelta
import datetime
import calendar
import pandas as pd
import pickle

outdir= '/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'
procdir=outdir
ddir= '/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'

model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}


#read in different files
def open_ds(var,case,esm='cesm',stream='h0',model='cam'):
    if esm=='cesm':
        if case[:4]=='hist':
            comp='BHIST'
            tseriesdir = ddir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-hist-'+case.split('-')[1]+'/' 
        else:
            comp='BSSP126'
            tseriesdir = ddir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'/' 

        # define filename
        fn = var+'_b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'.'+ model + '.' + stream + '.nc'
    else:
        ens='i'+case.split('-')[1][-1]
        case=case.split('-')[0]

        if case=='hist':
            case='histctl'
            tspan='1980-2014'
        else:
            tspan='2015-2099'
        tseriesdir = ddir + esm+'/'+case+'/'
        # define filename
        if stream=='h0':
            fn = var+'_monthly_'+esm+'_'+case+'_'+ens+'_no-dynveg_'+tspan+ '.nc'
        elif stream=='h2':
            fn = var+'_3hr_'+esm+'_'+case+'_'+ens+'_no-dynveg_'+tspan+ '.nc'
        elif stream=='h1':
            fn = var+'_'+esm+'_'+case+'_'+ens+'_no-dynveg_'+tspan+ '.nc'
    # check if variable timeseries exists and open variable as data array
    if not os.path.isfile(tseriesdir + fn):
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
            ds['time']=pd.to_datetime(ds['time'], format='%Y%m%d') 
        elif esm=='ecearth':
            ds = xr.open_dataset(tseriesdir+fn)
        
        return ds
def open_ds_mean(var,case,esm='cesm',model='cam'):
    # define filename
    if esm=='cesm':
        if case[:4]=='hist':
            comp='BHIST'
            savedir = ddir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-hist-'+case.split('-')[1] + '/mean/'
            fn = case.split('-')[0][:4]+'.'+model+'.mean_'+var + '.'+case[-3:] + '.nc'
        else:
            comp='BSSP126'
            savedir = ddir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case + '/mean/'
            fn = case.split('-')[0]+'.'+model+'.mean_'+var + '.'+case[-3:] + '.nc'
    else:
        print(case.split('-'))
        ens='i'+case.split('-')[1][-1]
        case=case.split('-')[0]
        if case=='hist':
            case='histctl'
            tspan='1980-2014'
        else:
            tspan='2015-2099'
        tseriesdir = ddir + esm+'/'+case+'/'
        # define filename
        if stream=='h0':
            fn = var+'_monthly_'+esm+'_'+case+'_'+ens+'_no-dynveg_'+tspan+ '.nc'
        elif stream=='h2':
            fn = var+'_3hr_'+esm+'_'+case+'_'+ens+'_no-dynveg_'+tspan+ '.nc'
        elif stream=='h1':
            fn = var+'_'+esm+'_'+case+'_'+ens+'_no-dynveg_'+tspan+ '.nc'
    # check if variable timeseries exists and open variable as data array
    if not os.path.isfile(savedir + fn):
        print(fn + ' does not exists in ')
        print(savedir)
        return
    else:     
        if esm=='cesm':
            ds = xr.open_dataset(tseriesdir+fn)
            ds['time']=ds['time'].astype("datetime64[ns]")

        elif esm=='mpiesm':
           # ds = xr.open_dataset(savedir+fn)
            #print(ds.time.attrs['units'])
            ds = xr.open_dataset(tseriesdir+fn)
            ds['time']=pd.to_datetime(ds['time'], format='%Y%m%d') 
        elif esm=='ecearth':
            ds = xr.open_dataset(tseriesdir+fn)
        return ds



def open_da_mean(var, case, esm='cesm',model='cam'):
    ds = open_ds_mean(var, case, esm, model)
    da = ds[var]
    return da


def open_da(var, case, esm='cesm', stream='h0', model='cam',isxtrm=False):
    if isxtrm:
        ds = open_ds_xtrm(var, case, esm, stream, model)
    else:
        ds = open_ds(var, case, esm, stream, model)
    da = ds[var]
    #shift time by one (to fix index error in preprocessing)
   # if stream=='h0':
   #     t=xr.CFTimeIndex(da.time.get_index('time') - timedelta(days=31))
   # elif stream=='h1':
   #     t=xr.CFTimeIndex(da.time.get_index('time') - timedelta(days=1))
   # elif stream=='h2':
   #     t=xr.CFTimeIndex(da.time.get_index('time') - timedelta(hours=3))
   # print(t)
    #da['time']=t
    return da



def open_da_delta(var, case, case_ref,esm, stream, model, ens=True, mode='mean',isxtrm=False): 
    """ open and caluclate the difference between the ensemble means of two members """
    # Load the two datasets
    if ens == True:
        da_res = open_da_ens(var,case=case, esm=esm, n_ens=3, stream=stream, model=model, mode = mode,isxtrm=isxtrm)
        da_ctl = open_da_ens(var,case=case_ref, esm=esm, n_ens=3, stream=stream, model=model, mode = mode,isxtrm=isxtrm)     
    else: # open single simulation
        da_res = open_da(var,case=case,esm=esm, model=model,isxtrm=isxtrm)
        da_ctl = open_da(var,case=case_ref,esm=esm, model=model,isxtrm=isxtrm)

    # calculate difference and update attributes
    print(da_ctl.time)
    print(da_res.time)
    da1=da_res
    da2=da_ctl
    #da1=extract_anaperiod(da_res, esm,stream,isxtrm=isxtrm)
    #if case_ref=='hist':
    #    da2=extract_anaperiod_hist(da_ctl, esm,stream,isxtrm=isxtrm)
    #else:
    #    da2=extract_anaperiod(da_ctl, esm,stream,isxtrm=isxtrm)
    #print(da1.time)
    #print(da2.time)
    if isxtrm:
        da2['year']=da1.year
    else:
        da2['time']=da1.time
    da_delta =  da1-da2

    #da_delta.attrs['long_name'] = '$\Delta$ '+ da_ctl.long_name
    #da_delta.name = '$\Delta$ '+ da_ctl.name

    return da_delta

def open_da_delta_mean(var, case, case_ref, esm, model, ens=True, mode='mean'): 
    """ open and caluclate the difference between the ensemble means of two members """
    
    # Load the two datasets
    if ens == True:
        da_res = open_da_ens_mean(var,case=case, esm=esm, n_ens=3, model=model, mode = mode)
        da_ctl = open_da_ens_mean(var,case=case_ref, esm=esm, n_ens=3, model=model, mode = mode)     
    else: # open single simulation
        da_res = open_da_mean(var,case=case,esm=esm, model=model)
        da_ctl = open_da_mean(var,case=case_ref,esm=esm, model=model)

    # calculate difference and update attributes
    da_delta = da_res - da_ctl

    return da_delta

def open_da_ens(var, case, esm, n_ens, stream='h0', model='cam', mode='mean',isxtrm=False):
    # loop over ensemble members
    print(case)
    for i in range(1,n_ens+1): 
        case_name = case+'-i308.00'+str(i) 
        
        da = open_da(var, case=case_name, esm=esm, stream=stream, model=model,isxtrm=isxtrm)
        print(da)
        if case=='hist':
            da=extract_anaperiod_hist(da, esm,stream=stream,isxtrm=isxtrm)
        else:
            da=extract_anaperiod(da, esm,stream=stream,isxtrm=isxtrm)
        if i==1: 
            da_concat = da 
        else: 
            if isxtrm:
                da['year']=da_concat.year
            else:
                da['time']=da_concat.time
            da_concat = xr.concat((da_concat, da), dim='ens_member')

    # different output options
    
    # return ensemble mean
    if mode == 'mean': 
        return da_concat.mean(dim='ens_member', keep_attrs='True')
    
    # standard deviation
    if mode == 'std': 
        return da_concat.std(dim='ens_member', keep_attrs='True')
    
    # the full ensemble with dim (ens_member)
    if mode == 'all':
        return da_concat

def open_da_ens_mean(var, case, esm, n_ens, model='cam', mode='mean'):
    ens_list=['000','001','002','003']
    # loop over ensemble members
    for i in range(1,n_ens+1): 
        case_name = case+'-i308.00'+str(i) 

        da = open_da_mean(var, case=case_name, esm=esm, model=model)

        if i==1: 
            da_concat = da 
        else: 
            da_concat = xr.concat((da_concat, da), dim='ens_member')

    # different output options
    
    # return ensemble mean
    if mode == 'mean': 
        return da_concat.mean(dim='ens_member', keep_attrs='True')
    
    # standard deviation
    if mode == 'std': 
        return da_concat.std(dim='ens_member', keep_attrs='True')
    
    # the full ensemble with dim (ens_member)
    if mode == 'all':
        return da_concat
    
    
# save dataset as nc in postprocessing dir for extremes   
def save_da_xtrm(da,var,case, esm, block,comp,ens):
    if esm=='cesm':
        savedir = procdir +  'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ens + '/tseries/'
        # define filename
        fn = case + '.'+ model[block] + '.' + var + '.' + ens +'.nc'
    else:
        case=case.split('_')[0]
        if case=='hist':
            case='histctl'
        savedir = ddir + esm+'/'+case+'/tseries/'
        # define filename
        fn = case + '.'+ model[block] + '.' + var + '.' + ens +'.nc'
    # check if variable timeseries exists and open variable as data array
    if os.path.isfile(savedir + fn):
        print(fn + ' already exists')
    else: 
        da.to_dataset().to_netcdf(savedir+fn)
    return

def extract_anaperiod_hist(da,esm,stream,isxtrm=False):
    """Extract analysis period out of data-array (last 30 years)"""
    # number of spin up years
    if esm=='cesm':  ##cesm contains 1 timestep in the next year which should not be taken into account
        if isxtrm:
            if da.name in ['TREFHTdaymax']:
                if len(da.shape)>3:
                    da = da[:,-30*365-1:-1,:,:]
                else:
                    # spin up with monthly timestep
                    da = da[-30*365-1:-1,:,:]
            else:
                if len(da.shape)>3:
                    da = da[:,-30-1:-1,:,:]
                else:
                    # spin up with monthly timestep
                    da = da[-30-1:-1,:,:]
        else:
            if len(da.shape)>3:
                if stream == 'h2' : # this option still to test
                    # 3 hourly timesteps
                    da = da[:,-30*365*8-1:-1,:,:]
                else:
                    # spin up with monthly timestep
                    da = da[:,-20*12-1:-1,:,:]
            else:
                if stream == 'h2' : # this option still to test
                    # 3 hourly timesteps
                    da = da[-20*365*8-1:-1,:,:]
                else:
                    # spin up with monthly timestep
                    da = da[-20*12-1:-1,:,:]
    else:
        if len(da.shape)>3:
            if stream == 'h2' : # this option still to test
                    # 3 hourly timesteps
                    da = da[:,-30*365*8:,:,:]
            else:
                    # spin up with monthly timestep
                    da = da[:,-20*12:,:,:]
        else:
            if stream == 'h2' : # this option still to test
                    # 3 hourly timesteps
                    da = da[-30*365*8:,:,:]
            else:
                    # spin up with monthly timestep
                    da = da[-20*12:,:,:]
    return da

def extract_anaperiod(da, esm,stream,isxtrm=False): 
    """Extract analysis period out of data-array (2040-2060)"""
    # number of spin up years
    if esm=='cesm':  ##cesm contains 1 timestep in the next year which should not be taken into account
        print(da.shape)
        if isxtrm:
            if da.name in ['TREFHTdaymax']:
                if len(da.shape)>3:
                    da = da[:,-60*365-1:-40*365-1,:,:]
                else: 
                    # spin up with monthly timestep
                    da = da[-60*365-1:-40*365-1,:,:]
            else:
                if len(da.shape)>3:
                    da = da[:,-60*30-1:-40*30-1,:,:]
                else: 
                    # spin up with monthly timestep
                    da = da[-60*30-1:-40*30-1,:,:]
        else:
            if len(da.shape)>3:
                if stream == 'h2' : # this option still to test 
                    # 3 hourly timesteps
                    da = da[:,-60*365*8-1:-40*365*8-1,:,:]
                else: 
                    # spin up with monthly timestep
                    da = da[:,300:540,:,:]
            else:
                if stream == 'h2' : # this option still to test 
                    # 3 hourly timesteps
                    da = da[-60*365*8-1:-40*365*8-1,:,:]
                else: 
                    # spin up with monthly timestep
                    da = da[300:540,:,:]
    else:
        if len(da.shape)>3:
            if stream == 'h2' : # this option still to test 
                    # 3 hourly timesteps
                    da = da[:,-60*365*8:-40*365*8,:,:]
            else: 
                    # spin up with monthly timestep
                    da = da[:,300:540,:,:]
        else:
            if stream == 'h2' : # this option still to test 
                    # 3 hourly timesteps
                    da = da[8:-40*365*8,:,:]
            else: 
                    # spin up with monthly timestep
                    da = da[300:540,:,:]
    return da

# open dataset of extremes
def open_ds_xtrm(var,case,esm,stream='h0',model='cam'):
    if esm=='cesm':
        if case[:4]=='hist':
            comp='BHIST'
            tseriesdir = ddir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-hist-'+case.split('-')[1]+'/tseries/' 
            fn = case.split('-')[0][:4]+'.'+model+'.'+var + '.'+case[-3:] + '.nc'
        else:
            comp='BSSP126'
            tseriesdir = ddir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'/tseries/' 
            fn = case.split('-')[0]+'.'+model+'.'+var + '.'+case[-3:] + '.nc'
    else:
        case=case.split('_')[0]
        if case=='hist':
            case='histctl'
        savedir = ddir + esm+'/'+case+'/tseries/'
        # define filename
        fn = case + '.'+ model[block] + '.' + var + '.' + ens +'.nc'
    if not os.path.isfile(tseriesdir + fn):
        print(fn + ' does not exists in ')
        print(tseriesdir)
        return
    else: 
        
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
    return ds
    
    


# check if dataset of extremes exists
def exist_da_xtrm(var,case, esm, block,comp,ens):
    if esm=='cesm':
        savedir = outdir +  'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ens + '/tseries/'
        # define filename
        fn = case + '.'+ model[block] + '.' + var + '.' + ens +'.nc'
    else:
        case=case.split('_')[0]
        if case=='hist':
            case='histctl'
        savedir = ddir + esm+'/'+case+'/tseries/'
        # define filename
        fn = case + '.'+ model[block] + '.' + var + '.' + ens +'.nc'
    # check if variable timeseries exists and open variable as data array
    if not os.path.isfile(savedir + fn): exists = False
    else: exists = True
        
    return exists

def conv_m_s_to_mm_day(da_in):

    if not da_in.attrs['units'] == 'mm/day':
        da_out = da_in * 86400000  
        # update attributes and change units
        da_out.attrs= da_in.attrs
        da_out.attrs['units'] = 'mm/day' 
    else: 
        da_out = da_in
    return da_out

#compute average of extremes
def mean_xtrm(var,case,esm,block,ens):
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
   # check if var is already existing
    if  exist_da_xtrm(var,case, esm,block,comp,ens):
        # open da with daily data
        case_name = case+'-i308.'+ens 
        da = open_da(var, case_name, esm,stream='h0', model=model[block],isxtrm=True)
        da=extract_anaperiod(da, esm,stream='h0',isxtrm=True)
        # calculate mean
        da_mean= da.mean('year')

        da_mean.name = var
        da_mean.attrs['long_name'] = 'Mean '+da.long_name
        if esm=='cesm':
            savedir = procdir +  'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ens + '/mean/'
            # define filename
            fn = case + '.'+ model[block] + '.mean_' + var + '.' + ens +'.nc'
        else:
            case=case.split('_')[0]
            if case=='hist':
                case='histctl'
            savedir = ddir + esm+'/'+case+'/mean/'
            # define filename
            fn = case + '.'+ model[block] + '.mean_' + var + '.' + ens +'.nc'
        # check if variable timeseries exists and open variable as data array
        if os.path.isfile(savedir + fn):
            print(fn + ' already exists')
        else:
            da_mean.to_dataset().to_netcdf(savedir+fn)
    else:
        print('extreme does not exist')
    return

#compute averages
def mean_var(var,case,esm,stream,block,ens):
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
   # check if var is already existing
    # open da with daily data
    case_name = case+'-i308.'+ens 
    da = open_da(var,case_name,esm,stream,model[block])
    da=extract_anaperiod(da,esm, stream=stream)
    # calculate mean
    da_mean= da.mean('time')
    da_mean.name = var
    da_mean.attrs['long_name'] = 'Mean '+da.long_name
    if esm=='cesm':
        savedir = procdir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ens + '/mean/'
        # define filename
        fn = case + '.'+ model[block] + '.mean_' + var + '.' + ens +'.nc'
    else:
        case=case.split('_')[0]
        if case=='hist':
            case='histctl'
        savedir = ddir + esm+'/'+case+'/mean/'
        # define filename
        fn = case + '.'+ model[block] + '.mean_' + var + '.' + ens +'.nc'
    # check if variable timeseries exists and open variable as data array
    if os.path.isfile(savedir + fn):
        print(fn + ' already exists')
    else:
        da_mean.to_dataset().to_netcdf(savedir+fn)
    return

def comp_PRECT(case,esm='cesm', ens='001'):
    if esm=='cesm':
        if case=='hist':
            comp='BHIST'
        else:
            comp='BSSP126'
       # check if var is already existing
        # open da with daily data
        case_name = case+'-i308.'+ens 
        da1 = open_da('PRECC',case_name, stream='h0',model=model['atm'])
        da2 = open_da('PRECL',case_name, stream='h0',model=model['atm'])
        da=da1+da2
    else:
        case_name = case+'-i308.'+ens 
        da = open_da('pr',case_name, esm=esm, stream='h0',model=model['atm'])
    
    # calculate mean
    da_e=extract_anaperiod(da, esm,stream='h0')

    da_mean= da_e.mean('time')
    da.name = 'PRECT'
    da_mean.name = 'PRECT'

    if esm=='cesm':
        savedir = procdir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ens +'/'
        # define filename
        block='atm'
        fn_mean = case + '.'+ model[block] + '.mean_PRECT.' + ens +'.nc'
        fn='PRECT_b.e213.'+comp+'_BPRP.f09_g17.esm-'+case + '-i308.'+ens+'.cam.h0.nc'
    else:
        return
        ###TO FILL IN
    # check if variable timeseries exists and open variable as data array
    if os.path.isfile(savedir +'/mean/'+ fn_mean):
        print(fn_mean + ' already exists')
    else:
        da_mean.to_dataset().to_netcdf(savedir+'/mean/'+fn_mean)
    if os.path.isfile(savedir + fn):
        print(fn + ' already exists')
    else:
        da.to_dataset().to_netcdf(savedir+fn)
    return

def comp_PRECT_day(case,ens):
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
   # check if var is already existing
    # open da with daily data
    case_name = case+'-i308.'+ens 
    model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}
    da = open_da('PRECT',case_name, stream='h2',model=model['atm'])
    # calculate mean
    
    da_day=da.resample(time='D').sum(keep_attrs=True)
    
    da_day.name = 'PRECT'
    savedir = procdir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ens +'/'
    # define filename
    block='atm'

    fn='PRECT_b.e213.'+comp+'_BPRP.f09_g17.esm-'+case + '-i308.'+ens+'.cam.h1.nc'
    # check if variable timeseries exists and open variable as data array
  
    if os.path.isfile(savedir + fn):
        print(fn + ' already exists')
    else:
        da_day.to_dataset().to_netcdf(savedir+fn)
    return

# process TXx: calculate and save annual maximum of maxdaytime temperature 
def proc_TXx(var_or, case, esm,block,ens):
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # define new variable name
    if var_or=='TS' or var_or=='ts':
        var = 'TXx'
    elif var_or=='TREFHT' or var_or=='tas':
        var='TRXx'
    else:
        print('incorrect variable')
        return

    # check if var is already existing
    if  exist_da_xtrm(var,case, esm,block,comp,ens):
        print(var +' already exists')
    else: # do calculations
        case_name = case+'-i308.'+ens 
        # open da with daily data
        da = open_da(var_or,case_name, esm, stream='h2', model=model[block])
        
        # calculate maximum per year
        da_xtrm= da.groupby('time.year').max(keep_attrs=True)
        da_xtrm.name = var
        da_xtrm.attrs['long_name'] = 'Annual maximum of '+da.long_name
        
        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, esm=esm, block=block,comp=comp,ens=ens)
# process TXx: calculate and save annual maximum of maxdaytime temperature 
def proc_Tdaymax(var_or, case, block,ens):
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # define new variable name
    if var_or=='TS':
        var = 'TSdaymax'
    elif var_or=='TREFHT':
        var='TREFHTdaymax'
    else:
        print('incorrect variable')
        return
    model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}

    # check if var is already existing
    if  exist_da_xtrm(var,case, block,comp,ens):
        print(var +' already exists')
    else: # do calculations
        case_name = case+'-i308.'+ens 
        # open da with daily data
        da = open_da(var_or,case_name, stream='h2', model=model[block])
        # calculate maximum per year
        da_xtrm=da.resample(time='D').max(keep_attrs=True)
        da_xtrm.name = var
        da_xtrm.attrs['long_name'] = 'daily maximum of '+da.long_name
        
        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens)

# process TNn: calculate and save annual maximum of maxdaytime temperature 
def proc_TNn(var_or, case, block,ens):
    
    # define new variable name
    if var_or=='TS':
        var = 'TNn'
    elif var_or=='TREFHT':
        var='TRNn'
    else:
        print('incorrect variable')
        return
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}

    # check if var is already existing
    if  exist_da_xtrm(var,case=case, block=block,comp=comp,ens=ens):
        print(var +' already exists')
    else: # do calculations
        case_name = case+'-i308.'+ens 
        # open da with daily data
        da = open_da(var_or,case_name, stream='h2', model=model[block])

        # calculate minimum per year
        da_xtrm = da.groupby('time.year').min(keep_attrs=True)

        da_xtrm.name = var
        da_xtrm.attrs['long_name'] = 'Annual minimum of '+da.long_name
        
        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens)

        
# calculate 99th percentile of max daytime temperatures 
def proc_TX99(var_or, case, block,ens):
    
    # define new variable name
    if var_or=='TS':
        var = 'TX99'
    elif var_or=='TREFHT':
        var='TRX99'
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # check if var is already existing
    if  exist_da_xtrm(var,case=case, block=block,comp=comp,ens=ens):
        print(var +' already exists')
    else: # do calculations
        case_name = case+'-i308.'+ens 
        # open da with daily data
        da = open_da(var_or,case_name, stream='h2', model=model[block])
        da_day=da.resample(time='D').max(keep_attrs=True)
        # calculate maximum per year
        da_xtrm  = da_day.quantile(0.99, dim=('time'))        

        da_xtrm.name = var
        da_xtrm.attrs['long_name'] = '99th percentile of daily '+da.long_name
        da_xtrm.attrs['units'] = 'K'

        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens)        

        
# calculate 1st percentile of min nighttime temperatures 
def proc_TN01(var_or, case, block,ens):
    
    # define new variable name
    if var_or=='TS':
        var = 'TN01'
    elif var_or=='TREFHT':
        var='TRN01'
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # check if var is already existing
    if  exist_da_xtrm(var,case=case, block=block,comp=comp,ens=ens):
        print(var +' already exists')
    else: # do calculations
        case_name = case+'-i308.'+ens 
        # open da with daily data
        da = open_da(var_or,case_name, stream='h2', model=model[block])
        da_day = da.resample(time='D').min(keep_attrs=True)
        
        # calculate maximum per year
        da_xtrm  = da_day.quantile(0.01, dim=('time'))
        
        da_xtrm.name = var
        da_xtrm.attrs['long_name'] = '1st percentile of daily '+da.long_name
        da_xtrm.attrs['units'] = 'K'
        
        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens)   

def proc_TN10(var_or, case, block,ens):
    """calculcate and save cold days 10pctl of days within period """
    # define new variable name
    if var_or=='TS':
        var = 'TN10'
    elif var_or=='TREFHT':
        var='TRN10'
    
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # check if var is already existing
    if  exist_da_xtrm(var,case=case, block=block,comp=comp,ens=ens):
        print(var +' already exists')
    else: # do calculations
        case_name = case+'-i308.'+ens 
        # open da with daily data
        da = open_da(var_or,case_name, stream='h2', model=model[block])
        da_day = da.resample(time='D').min(keep_attrs=True)
        
        # calculate maximum per year
        da_xtrm  = da_day.quantile(0.1, dim=('time'))
        
        da_xtrm.name = var
        da_xtrm.attrs['long_name'] = '10th percentile of daily'+da.long_name
        da_xtrm.attrs['units'] = 'K'
        
        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens)        

def proc_TX90(var_or, case, block,ens):
    """calculcate and save warm days 90th pctl of days within period """

    # define new variable name
    if var_or=='TS':
        var = 'TX90'
    elif var_or=='TREFHT':
        var='TRX90'
    
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # check if var is already existing
    if  exist_da_xtrm(var,case=case, block=block,comp=comp,ens=ens):
        print(var +' already exists')
    else: # do calculations
        case_name = case+'-i308.'+ens 
        # open da with daily data
        da = open_da(var_or,case_name, stream='h2', model=model[block])
        da_day = da.resample(time='D').max(keep_attrs=True)
        
        # calculate maximum per year
        da_xtrm  = da_day.quantile(0.9, dim=('time'))
        

        da_xtrm.name = var
        da_xtrm.attrs['long_name'] = '90th percentile of daily'+da.long_name
        da_xtrm.attrs['units'] = 'K'

        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens)        
    
def proc_Rx1day(var_or, case, block,ens):
    """process Rx1day: calculate annual maximum 1 day precipitation"""
    # define new variable name
    var = 'Rx1day'
    model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}
    
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # check if var is already existing
    if  exist_da_xtrm(var,case=case, block=block,comp=comp,ens=ens):
        print(var +' already exists')
    else: # do calculations
        case_name = case+'-i308.'+ens 
        # open da with daily data
        da = open_da(var_or,case_name, stream='h2', model=model[block])
        da_day = da.resample(time='D').sum(keep_attrs=True)
        da_day[:]=da_day.values*3
        # calculate maximum per year
        da_xtrm= da.groupby('time.year').max(keep_attrs=True)

        da_xtrm.name = var
        da_xtrm.attrs['long_name'] = 'Annual maximum of '+da.long_name
        
        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens)        

def proc_R05(var_or, case, block,ens):
    """calculcate and save 5th pctl of monthly precip: drought months  """
    # define new variable name
    var = 'R05'
    
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # check if var is already existing
    if  exist_da_xtrm(var,case=case, block=block,comp=comp,ens=ens):
        print(var +' already exists')
    else: # do calculations
        # open da with daily data(var, case, n_ens, stream='h0', model='cam', mode='mean',isxtrm=False)
        da = open_da_ens(var_or,case=case, n_ens=3, stream='h0', model='cam', mode='all')
        da_lumped = da.stack(dim=("ens_member", "time"))
        # calculate quantile over months

        da_xtrm = da_lumped.quantile(0.05, dim=('dim'))
        da_xtrm.name = var

        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens)   
    return
       
def proc_R95(var_or, case, block,ens):
    """calculcate and save 95th pctl of monthly precip: wet months  """
    # define new variable name
    var = 'R95'
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # check if var is already existing
    if  exist_da_xtrm(var,case=case, block=block,comp=comp,ens=ens):
        print(var +' already exists')
    else: # do calculations

        # open da with daily data
        da = open_da_ens(var_or,case=case, n_ens=3, stream='h0', model='cam', mode='all')
        da_lumped = da.stack(dim=("ens_member", "time"))
        # calculate quantile over months

        da_xtrm = da_lumped.quantile(0.95, dim=('dim'))
        da_xtrm.name = var

        # save variable into netcdf
        save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens)   
    return

 
def proc_TXx_monthly(var_or, case, block,ens):
    """ process TXx: calculate and save monthly maximum of daytime temperature and save in monthly folder"""
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # define new variable name
    if var_or=='TS':
        var = 'TXx_m'
    elif var_or=='TREFHT':
        var='TRXx_m'
    else:
        print('incorrect variable')
        return
    model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}
    # check if var is already existing
    case_name = case+'-i308.'+ens 
    # open da with daily data
    da = open_da(var_or,case_name, stream='h2', model=model[block])
    # calculate maximum per year
    da_xtrm = da.resample(time='M').max(keep_attrs=True)

    da_xtrm.name = var
    da_xtrm.attrs['long_name'] = 'Monthly maximum of '+da.long_name
    da_ex=extract_anaperiod(da_xtrm, esm,stream='h0')

    da_mean= da_ex.mean('time')
    da_mean.name = var

    savedir = procdir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ens +'/'
    # define filename

    fn_mean = case + '.'+ model[block] + '.mean_'+var+'.' + ens +'.nc'
    fn=var+'_b.e213.'+comp+'_BPRP.f09_g17.esm-'+case + '-i308.'+ens+'.cam.h0.nc'
    # check if variable timeseries exists and open variable as data array
    if os.path.isfile(savedir +'/mean/'+ fn_mean):
        print(fn_mean + ' already exists')
    else:
        da_mean.to_dataset().to_netcdf(savedir+'/mean/'+fn_mean)
    if os.path.isfile(savedir + fn):
        print(fn + ' already exists')
    else:
        da_xtrm.to_dataset().to_netcdf(savedir+fn)
    return
        

# process TXx: calculate and save annual maximum of maxdaytime temperature 
def proc_TNn_monthly(var_or, case, block,ens):
    """ process TNn: calculate and save monthly minimum of nighttime temperature and save in monthly folder"""
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # define new variable name
    if var_or=='TS':
        var = 'TNn_m'
    elif var_or=='TREFHT':
        var='TRNn_m'
    else:
        print('incorrect variable')
        return
    model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}
    # check if var is already existing
    case_name = case+'-i308.'+ens 
    # open da with daily data
    da = open_da(var_or,case_name, stream='h2', model=model[block])
    # calculate maximum per year
    da_xtrm = da.resample(time='M').max(keep_attrs=True)

    da_xtrm.name = var
    da_xtrm.attrs['long_name'] = 'Monthly minimum of '+da.long_name
    da_ex=extract_anaperiod(da_xtrm, esm,stream='h0')

    da_mean= da_ex.mean('time')
    da_mean.name = var

    savedir = procdir + 'b.e213.'+comp+'_BPRP.f09_g17.esm-'+case+'-i308.'+ens +'/'
    # define filename

    fn_mean = case + '.'+ model[block] + '.mean_'+var+'.' + ens +'.nc'
    fn=var+'_b.e213.'+comp+'_BPRP.f09_g17.esm-'+case + '-i308.'+ens+'.cam.h0.nc'
    # check if variable timeseries exists and open variable as data array
    if os.path.isfile(savedir +'/mean/'+ fn_mean):
        print(fn_mean + ' already exists')
    else:
        da_mean.to_dataset().to_netcdf(savedir+'/mean/'+fn_mean)
    if os.path.isfile(savedir + fn):
        print(fn + ' already exists')
    else:
        da_xtrm.to_dataset().to_netcdf(savedir+fn)
    return
   
## 
# get da of CDD: annual number of consecutive dry days 
# because not possible to save as netcdf. 
def get_CDD(var_or, case,ens):
    if case=='hist':
        comp='BHIST'
    else:
        comp='BSSP126'
    # define new variable name
    var = 'CDD'
    block='atm'
    model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}
    case_name = case+'-i308.'+ens
    # open da with daily data and convert to mm/day
    da = open_da(var_or,case=case_name, stream='h1', model=model[block])
    da = 1000*3600*24*da

    # initialise np array to fill per year
    da_year_mean = da.groupby('time.year').mean()
    cdd_max_peryear = np.empty_like(da_year_mean.values)

    # Loop over grouped data per year 
    for i, da_year in enumerate(list(da.groupby('time.year'))): 
        # create empty np array to save boolean variables indication day is dry
        drydays = np.empty_like(da_year[1].values)
        drydays = da_year[1].values < 1

        # empty cdd matrix per year 
        cdd = np.empty_like(da_year[1].values)

        # loop over all days of the year
        for d in range(len(drydays)):
            if d > 0: 
                ddind = np.where(drydays[d,:,:])
                notddind = np.where(drydays[d,:,:]==False)

                cdd_curd = cdd[d,:,:] 
                cdd_prevd = cdd[d-1,:,:]

                cdd_curd[ddind] = cdd_prevd[ddind] +1
                cdd_curd[notddind] = 0

                cdd[d,:,:] = cdd_curd


        # define maximum of cdd of the year
        cdd_max_peryear[i,:,:] =  cdd.max(axis=0)

    # save into data array and update attributes
    da_xtrm = xr.DataArray(cdd_max_peryear, coords= da_year_mean.coords,
         dims=da_year_mean.dims)
    da_xtrm.name = 'CDD'
    da_xtrm.attrs['units'] = 'days'
    da_xtrm.attrs['long_name'] = 'Annual maximal number of consecutive dry days (PR>1 mm/day)'
    
    
    save_da_xtrm(da_xtrm,var,case=case, block=block,comp=comp,ens=ens) 
    return da_xtrm

def get_CDD_iv(var_or, case, block='atm'):
    
    # define new variable name
    var = 'CDD'
    
    # open da with daily data and convert to mm/day
    da = open_da(var_or,case=case, stream='h1', block=block,ens=ens)
    da = 1000*3600*24*da


    # initialise np array to fill per year
    da_year_mean = da.groupby('time.year').mean()
    cdd_max_peryear = np.empty_like(da_year_mean.values)

    # Loop over grouped data per year 
    for i, da_year in enumerate(list(da.groupby('time.year'))): 
        # create empty np array to save boolean variables indication day is dry
        drydays = np.empty_like(da_year[1].values)
        drydays = da_year[1].values < 1

        # empty cdd matrix per year 
        cdd = np.empty_like(da_year[1].values)

        # loop over all days of the year
        for d in range(len(drydays)):
            if d > 0: 
                ddind = np.where(drydays[d,:,:])
                notddind = np.where(drydays[d,:,:]==False)

                cdd_curd = cdd[d,:,:] 
                cdd_prevd = cdd[d-1,:,:]

                cdd_curd[ddind] = cdd_prevd[ddind] +1
                cdd_curd[notddind] = 0

                cdd[d,:,:] = cdd_curd


        # define maximum of cdd of the year
        cdd_max_peryear[i,:,:] =  cdd.max(axis=0)


    # save into data array and update attributes
    da_xtrm = xr.DataArray(cdd_max_peryear, coords= da_year_mean.coords,
         dims=da_year_mean.dims)

    da_xtrm.name = 'CDD'
    da_xtrm.attrs['units'] = 'days'
    da_xtrm.attrs['long_name'] = 'Annual maximal number of consecutive dry days (PR>1 mm/day)'

    return da_xtrm

# get CDD for every ensemble member and calculate ensemble mean
def get_CDD_ens(case_nores, case_res, n_ens): 
    
    for i in range(1,n_ens+1):
        
        print('Calculating CDDs for ensemble member '+ str(i))

        cdd_nores = get_CDD('PRECT',case_nores,ens='00'  +str(i))
        cdd_res   = get_CDD('PRECT',case_res,ens='00'  +str(i))
        
        if i==1: 
            cdd_nores_concat = cdd_nores 
            cdd_res_concat   = cdd_res 

        else: 
            cdd_nores_concat = xr.concat((cdd_nores_concat, cdd_nores), dim='ens_member')
            cdd_res_concat = xr.concat((cdd_res_concat, cdd_res), dim='ens_member')

    cdd_nores_ensmean = cdd_nores_concat.mean(dim='ens_member', keep_attrs='True')
    cdd_res_ensmean   = cdd_res_concat.mean(  dim='ens_member', keep_attrs='True')   

    # calculate delta and update variables
    # calculate difference and update attributes
    da1=extract_anaperiod(cdd_res_ensmean, esm,stream='h0',isxtrm=True)
    da2=extract_anaperiod(cdd_nores_ensmean, esm,stream='h0',isxtrm=True)
    da2['year']=da1.year

    cdd_delta = da1 - da2
    cdd_delta.attrs['long_name'] = '$\Delta$ '+ cdd_res.long_name
    cdd_delta.attrs['units'] = cdd_res.units
    cdd_delta.name = '$\Delta$ '+ cdd_res.name
    
    return cdd_delta


# ---------------------------------------------------------------------
# 5. Functions to calculate and plot statistical significance
# ---------------------------------------------------------------------

def calc_pval(da_delta_ens_all, isxtrm = False):
    """ calculate pvalue ing the two-sided, paired, non-parametric Wilcoxon signed rank test """
    
    from scipy.stats import wilcoxon

    # extreme values don't have dimension 'time', but 'year' instead
    if isxtrm : lumped = da_delta_ens_all.stack(time_ensmember=("ens_member", "year"))
    else : lumped = da_delta_ens_all.stack(time_ensmember=("ens_member", "time"))
    
    # wilcoxon signed rank test for every grid point
    ncells = len(lumped.stack(gridcell=('lat','lon')).transpose())    
    p_values = np.empty(ncells)  

    for ind,gridcell in enumerate(lumped.stack(gridcell=('lat','lon')).transpose().values):
        
        # for all zero values, wilcoxon will not work. Assign manually a p-value of 1 as result will by default be non significant. 
        # this is the case for the ocean grid cells for the land model variables. 
        if np.count_nonzero(gridcell) == 0: 
            p = np.nan
        else: 
            w, p = wilcoxon(gridcell)
        
        p_values[ind] = p

    p_values_2D = p_values.reshape(len(da_delta_ens_all['lat']), len(da_delta_ens_all['lon']))

    da_p_values = xr.DataArray(data=p_values_2D, coords=(da_delta_ens_all.coords['lat'],da_delta_ens_all.coords['lon']), dims=('lat','lon'))

    return da_p_values

#def add_statsign( da_delta_ens_all, ax=ax, isxtrm=False, alpha = 0.05): 
#    """Add hatching to plot, indicating statistical significance"""
#
#    # calculate pvalues
#    da_p_values = calc_pval(da_delta_ens_all, isxtrm)
#    
#    #levels = [0, 0.05, 1]
#    ax.contourf(da_p_values.lon, da_p_values.lat, da_p_values, levels = [0, alpha, 1], hatches=['....', ''], colors='none')
#    
#    return ax
    
def get_statsign_mask(da_delta_ens_all, isxtrm=False, alpha = 0.05): 
    """get a statistical significance mask at alpha from data array with all ensemble members """
    
    # calculate pvalues
    da_p_values = calc_pval(da_delta_ens_all, isxtrm)
     
    return da_p_values< alpha

def get_statsign_fieldsign_mask(da_delta_ens_all, isxtrm=False, alpha = 0.05): 
    """get a statistical significance mask at alpha from data array with all ensemble members
    and mask for grid cells with field significance (based on False Discovery Rate)"""
    
    # calculate pvalues for two sided wilcoxon test 
    da_p_values = calc_pval(da_delta_ens_all, isxtrm)
    
    # mask p-values for field significance with False Discovery Rate
    sig_FDR = calc_fdr(da_p_values, alpha, print_values=False)
    
    h_values = da_p_values< alpha
    
    # get 1 and 0 mask 
    h_values_fs = h_values.where(sig_FDR==1)
     
    # return boolean
    return h_values_fs > 0

def calc_fdr(p_values, alpha = 0.05, print_values=True):
    """ function to check field significance with False Discovery Rate
        Wilks 2006, J. of Applied Met. and Clim.
        p_val = P-values at every grid point,
        h_val = 0 or 1 at every grid point, depending on significance level
        code translated from R to python from Lorenz et al. (2016)
        https://github.com/ruthlorenz/stat_tests_correlated_climdata/blob/master/FStests/false_discovery_rate_package.R """

    h_values = p_values< alpha
    
    # where pvalues nan, set h_values also to nan
    h_values.where(p_values != np.nan, np.nan)

    # K sum of local p values
    K = (h_values != np.nan).sum().values.item()

    # put all p-values in 1D vector
    prob_1D = p_values.stack(z=('lat','lon'))

    # sort vector increasing
    p_sort = prob_1D.sortby(prob_1D)


    # create empty data arrays
    p = prob_1D * np.nan
    fdr = p_values*0+1
    sig_FDR = p_values*0+1

    # reject those local tests for which max[p(k)<=(siglev^(k/K)]
    for k in range(0,K):
        if (p_sort[k] <= alpha*(k/K)):
            p[k] = p_sort[k].values.item()
        else: 
            p[k] = 0

    p_fdr = p.max() 

    fdr = fdr.where(p_values <= p_fdr)
    sig_FDR = sig_FDR.where(np.logical_and(fdr==1, h_values==1))

    sig_pts = sig_FDR.sum(skipna=True).values

    if print_values: 
        print('False Discovery Rate for Field Significance')
        print("Field significance level: "+ str(alpha))
        print('Number of significant points: '+str(sig_pts))
        print('Total Number of Tests: '+str(K))

    return sig_FDR

def calc_bin_statistics(da_tobin, da_forbinning, nbins, resmask, da_type, var_tobin):
    """ function to calculate bin statistics for the whole ensemble """

    # initialise empty arrays for statistics
    da_bin_median_all = np.array([])
    da_bin_25pct_all = np.array([])
    da_bin_75pct_all = np.array([])

    # loop over ensemble members and save per ensemble member if is not existing yet
    for ensmem in range(0,n_ens): #n_ens

        print('binning ens member: ' + str(ensmem+1))

        # selec ensemble members and take reservoir mask
        da_forbinning_mem = da_forbinning[ensmem,:]
        da_tobin_mem = da_tobin[ensmem,:]

        # do binning
        da_binned_mem = da_tobin_mem.groupby_bins(da_forbinning_mem, nbins)

        # calculate bin statistics per member
        da_bin_median_mem = da_binned_mem.median().values
        da_bin_25pct_mem = da_binned_mem.quantile(0.25).values
        da_bin_75pct_mem = da_binned_mem.quantile(0.75).values

        bin_dict_mem = {'median': da_bin_median_mem, 'Q25' : da_bin_25pct_mem, 'Q75' : da_bin_75pct_mem}
            


        if ensmem == 0:
            da_bin_median_all = da_bin_median_mem
            da_bin_25pct_all = da_bin_25pct_mem
            da_bin_75pct_all = da_bin_75pct_mem
        else: 
            da_bin_median_all = np.vstack((da_bin_median_all,da_bin_median_mem))
            da_bin_25pct_all  = np.vstack((da_bin_25pct_all,da_bin_25pct_mem))
            da_bin_75pct_all  = np.vstack((da_bin_75pct_all, da_bin_75pct_mem))

        
    # calculate bin statistics for whole ensemble
    da_bin_median = da_bin_median_all.mean(axis=0)
    da_bin_25pct = da_bin_25pct_all.mean(axis=0) 
    da_bin_75pct = da_bin_75pct_all.mean(axis=0)
    return (da_bin_median,da_bin_25pct, da_bin_75pct)


def save_bins(var_to_bin, var_for_binning):
    
    nbins = 20
    # load all ens members
    # used to determine bins
    da_forbinning_res   = open_da_ens(var_for_binning, case_res,   stream = 'h1', mode='all').where(resmask).mean(dim=('lat','lon'))

    if var_to_bin == 'WBGT': 
        da_forbinning_res = da_forbinning_res[:,:-1,:,:]    
                
    # used for binning
    da_tobin_res   = open_da_ens(var_to_bin, case_res,   stream = 'h1', mode='all').where(resmask).mean(dim=('lat','lon'))
    
    # calculate bin statistics
    (da_bin_median_res  , da_bin_25pct_res  , da_bin_75pct_res)    =  calc_bin_statistics(da_tobin_res,   da_forbinning_res,   nbins, resmask, 'res', var_to_bin)  
   
    # do the same for no res
    da_forbinning_nores = open_da_ens(var_for_binning, case_nores, stream = 'h1', mode='all')
    da_tobin_nores = open_da_ens(var_to_bin, case_nores, stream = 'h1', mode='all')

    if var_to_bin == 'WBGT': 
        da_forbinning_nores = da_forbinning_nores[:,:-1,:,:]
                
    (da_bin_median_nores, da_bin_25pct_nores, da_bin_75pct_nores)  =  calc_bin_statistics(da_tobin_nores, da_forbinning_nores, nbins, resmask, 'nores', var_to_bin)    
    
    bindiff       = da_bin_median_res - da_bin_median_nores
    bindiff_25pct = da_bin_25pct_res - da_bin_25pct_nores
    bindiff_75pct = da_bin_75pct_res - da_bin_75pct_nores
    
    bindiff_dict = {'median' : bindiff, 'Q25' : bindiff_25pct, 'Q75' : bindiff_75pct}
    np.save(var_to_bin+'_binT.npy', bindiff_dict, allow_pickle = True)
    
def read_tseries_as_pickle(var,esm,landmask=True,model='cam',box=None):
    time_slice_1={'cesm':-10951,'mpiesm':-10957,'ecearth':-10957} ##only select last 30 years
    time_slice_2={'cesm':-1,'mpiesm':None,'ecearth':None}
    da_mask=None
    ds_TREFHT_hist=open_da(var,'hist',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:]
    ds_TREFHT_ctl=open_da(var,'futctl',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:]
    ds_TREFHT_sust=open_da(var,'futsust',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:]
    ds_TREFHT_ineq=open_da(var,'futineq',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:]
    if landmask:
        da_mask=xr.open_dataset('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/landmask_'+esm+'_no_ice.nc')['landmask']
        da_mask['lat']=ds_TREFHT_hist.lat
        da_mask['lon']=ds_TREFHT_hist.lon
        mask_name='_land'
    if box!=None:
        if box[0]==-20 & box[1]==20 & box[3]==360:
            mask_name='_trop'
        elif box[2]==10:
            mask_name='_trop_congo'
        ds_TREFHT_hist_1=open_da(var,'hist',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_hist.lat<box[1]).where(ds_TREFHT_hist.lat>box[0]).where(ds_TREFHT_hist.lon>box[2]).where(ds_TREFHT_hist.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_hist_2=open_da(var,'hist',esm,model,'002','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_hist.lat<box[1]).where(ds_TREFHT_hist.lat>box[0]).where(ds_TREFHT_hist.lon>box[2]).where(ds_TREFHT_hist.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_hist_3=open_da(var,'hist',esm,model,'003','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_hist.lat<box[1]).where(ds_TREFHT_hist.lat>box[0]).where(ds_TREFHT_hist.lon>box[2]).where(ds_TREFHT_hist.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_hist=xr.concat([ds_TREFHT_hist_1,ds_TREFHT_hist_2,ds_TREFHT_hist_3],'time')

        # Save the array with Pickle
        with open('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'+var+'_hist_'+esm+mask_name+'.pkl', 'wb') as f:
            pickle.dump(ds_TREFHT_hist, f)
            print('done')

        ds_TREFHT_ctl_1=open_da(var,'futctl',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_ctl.lat<box[1]).where(ds_TREFHT_ctl.lat>box[0]).where(ds_TREFHT_ctl.lon>box[2]).where(ds_TREFHT_ctl.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_ctl_2=open_da(var,'futctl',esm,model,'002','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_ctl.lat<box[1]).where(ds_TREFHT_ctl.lat>box[0]).where(ds_TREFHT_ctl.lon>box[2]).where(ds_TREFHT_ctl.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_ctl_3=open_da(var,'futctl',esm,model,'003','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_ctl.lat<box[1]).where(ds_TREFHT_ctl.lat>box[0]).where(ds_TREFHT_ctl.lon>box[2]).where(ds_TREFHT_ctl.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_ctl=xr.concat([ds_TREFHT_ctl_1,ds_TREFHT_ctl_2,ds_TREFHT_ctl_3],'time')

        # Save the array with Pickle
        with open('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'+var+'_futctl_'+esm+mask_name+'.pkl', 'wb') as f:
            pickle.dump(ds_TREFHT_ctl, f)
            print('done')

        ds_TREFHT_sust_1=open_da(var,'futsust',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_sust.lat<box[1]).where(ds_TREFHT_sust.lat>box[0]).where(ds_TREFHT_sust.lon>box[2]).where(ds_TREFHT_sust.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_sust_2=open_da(var,'futsust',esm,model,'002','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_sust.lat<box[1]).where(ds_TREFHT_sust.lat>box[0]).where(ds_TREFHT_sust.lon>box[2]).where(ds_TREFHT_sust.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_sust_3=open_da(var,'futsust',esm,model,'003','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_sust.lat<box[1]).where(ds_TREFHT_sust.lat>box[0]).where(ds_TREFHT_sust.lon>box[2]).where(ds_TREFHT_sust.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_sust=xr.concat([ds_TREFHT_sust_1,ds_TREFHT_sust_2,ds_TREFHT_sust_3],'time')

        # Save the array with Pickle
        with open('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'+var+'_futsust_'+esm+mask_name+'.pkl', 'wb') as f:
            pickle.dump(ds_TREFHT_sust, f)
            print('done')

        ds_TREFHT_ineq_1=open_da(var,'futineq',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_ineq.lat<box[1]).where(ds_TREFHT_ineq.lat>box[0]).where(ds_TREFHT_ineq.lon>box[2]).where(ds_TREFHT_ineq.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_ineq_2=open_da(var,'futineq',esm,model,'002','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_ineq.lat<box[1]).where(ds_TREFHT_ineq.lat>box[0]).where(ds_TREFHT_ineq.lon>box[2]).where(ds_TREFHT_ineq.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_ineq_3=open_da(var,'futineq',esm,model,'003','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).where(ds_TREFHT_ineq.lat<box[1]).where(ds_TREFHT_ineq.lat>box[0]).where(ds_TREFHT_ineq.lon>box[2]).where(ds_TREFHT_ineq.lon<box[3]).mean(['lat','lon'])
        ds_TREFHT_ineq=xr.concat([ds_TREFHT_ineq_1,ds_TREFHT_ineq_2,ds_TREFHT_ineq_3],'time')

        # Save the array with Pickle
        with open('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'+var+'_futineq_'+esm+mask_name+'.pkl', 'wb') as f:
            pickle.dump(ds_TREFHT_ineq, f)
            print('done')

    else:
        ds_TREFHT_hist_1=open_da(var,'hist',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_hist_2=open_da(var,'hist',esm,model,'002','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_hist_3=open_da(var,'hist',esm,model,'003','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_hist=xr.concat([ds_TREFHT_hist_1,ds_TREFHT_hist_2,ds_TREFHT_hist_3],'time')

        # Save the array with Pickle
        with open('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'+var+'_hist_'+esm+mask_name+'.pkl', 'wb') as f:
            pickle.dump(ds_TREFHT_hist, f)
            print('done')

        ds_TREFHT_ctl_1=open_da(var,'futctl',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_ctl_2=open_da(var,'futctl',esm,model,'002','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_ctl_3=open_da(var,'futctl',esm,model,'003','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_ctl=xr.concat([ds_TREFHT_ctl_1,ds_TREFHT_ctl_2,ds_TREFHT_ctl_3],'time')

        # Save the array with Pickle
        with open('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'+var+'_futctl_'+esm+mask_name+'.pkl', 'wb') as f:
            pickle.dump(ds_TREFHT_ctl, f)
            print('done')

        ds_TREFHT_sust_1=open_da(var,'futsust',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_sust_2=open_da(var,'futsust',esm,model,'002','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_sust_3=open_da(var,'futsust',esm,model,'003','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_sust=xr.concat([ds_TREFHT_sust_1,ds_TREFHT_sust_2,ds_TREFHT_sust_3],'time')

        # Save the array with Pickle
        with open('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'+var+'_futsust_'+esm+mask_name+'.pkl', 'wb') as f:
            pickle.dump(ds_TREFHT_sust, f)
            print('done')

        ds_TREFHT_ineq_1=open_da(var,'futineq',esm,model,'001','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_ineq_2=open_da(var,'futineq',esm,model,'002','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_ineq_3=open_da(var,'futineq',esm,model,'003','h1')[time_slice_1[esm]:time_slice_2[esm],:,:].where(da_mask).mean(['lat','lon'])
        ds_TREFHT_ineq=xr.concat([ds_TREFHT_ineq_1,ds_TREFHT_ineq_2,ds_TREFHT_ineq_3],'time')

        # Save the array with Pickle
        with open('/dodrio/scratch/projects/2022_200/project_output/bclimate/sdeherto/postprocessing/'+var+'_futineq_'+esm+mask_name+'.pkl', 'wb') as f:
            pickle.dump(ds_TREFHT_ineq, f)
            print('done')
