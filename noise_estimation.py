# -*- coding: utf-8 -*-
"""
Estimate noise from raw Galion data using the ACF method
"""

import os
cd=os.path.dirname(__file__)
import warnings
from datetime import datetime
from scipy import stats
from matplotlib import pyplot as plt
import glob
import xarray as xr
import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.gridspec as gridspec

plt.close('all')
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

warnings.filterwarnings('ignore')

#%% Inputs
sources=['C:/Users/sletizia/Downloads/Lidar_data/Lidar_data']
pattern = re.compile(r'^\d{2}$')#pattern of folder names

#data format
skip=6#lines to skip
num_gates=34#number of gates
i_rws=1#column with velocity
i_snr=2#column with intensity
i_time=3#column with time

#noise estimation
N_lags=50#number of lags of ACF
max_nsi=0.5#maximum ratio of standard deviation standard deviation to mean standard deviation (os non-stationarity index, NSI)
DT_nsi=600# time window to evalue NSI
DT=600#[m] averaging time
min_time_bins=5#number of time bins to check non-stationarity
p_value=0.05#p_value for bootstrap
bins_snr=np.arange(-30.5,10.6)#[dB] bins in snr

#qc
N_range_excl=6#number of range gates to exclude
min_noise=10**-10 #[m/s] minimum noise level
max_failed=1#max number of failed noise

#%% Functions
def vstack(a,b):
    '''
    Stack vertically vectors
    '''
    if len(a)>0:
        ab=np.vstack((a,b))
    else:
        ab=b
    return ab   

def filt_mean(x,perc_lim=[5,95]):
    '''
    Mean with percentile filter
    '''
    x_filt=x.copy()
    x_filt[x_filt<np.nanpercentile(x_filt,perc_lim[0])]=np.nan
    x_filt[x_filt>np.nanpercentile(x_filt,perc_lim[1])]=np.nan    
    return np.nanmean(x)

def filt_BS_mean(x,p_value,M_BS=100,min_N=10,perc_lim=[5,95]):
    '''
    Mean with percentile filter and bootstrap
    '''
    x_filt=x.copy()
    x_filt[x_filt<np.nanpercentile(x_filt,perc_lim[0])]=np.nan
    x_filt[x_filt>np.nanpercentile(x_filt,perc_lim[1])]=np.nan    
    x=x[~np.isnan(x)]
    
    if len(x)>=min_N:
        x_BS=bootstrap(x,M_BS)
        mean=np.mean(x_BS,axis=1)
        BS=np.nanpercentile(mean,p_value)
    else:
        BS=np.nan
    return BS

def bootstrap(x,M):
    '''
    Bootstrap sample drawer
    '''
    i=np.random.randint(0,len(x),size=(M,len(x)))
    x_BS=x[i]
    return x_BS

#%% Initialization

#zeroing
ACF_all=[]
snr_all=[]

#%% Main
for s in sources:
    matched_folders=[]
    for dirpath, dirnames, filenames in os.walk(s):
       
        for dirname in dirnames:
            print(f'Scanning folder {dirname}')
            if pattern.match(dirname):
                matched_folders.append(os.path.join(dirpath, dirname))
    
    for m in matched_folders:
    
        files=glob.glob(os.path.join(m,'*scn'))
        print(f'{len(files)} found in {m}')
        
        snr_avg=(bins_snr[1:]+bins_snr[:-1])/2

        #zeroing
        tnum=np.zeros(len(files))
        rws=np.zeros((num_gates,len(files)))+np.nan
        snr=np.zeros((num_gates,len(files)))+np.nan
        rws_std=[]
        snr_std=[]
        
        #read text file
        i_f=0
        for file in files:
            with open(file, "r") as f:
                data=f.readlines()[skip:]
                
            tstr=data[0].split('\t')[i_time]
            
            tnum[i_f]=(datetime.strptime(tstr,'%Y-%m-%d %H:%M:%S.%f')-datetime(1970, 1, 1)).total_seconds()
            
            i_r=0
            for d in data:
                d_split=d.split('\t')
                rws[i_r,i_f]=d_split[i_rws]
                snr[i_r,i_f]=np.log10(np.float64(d_split[i_snr])-1)*10
                
                i_r+=1
            i_f+=1
        
        rws[:N_range_excl,:]=np.nan
        snr[:N_range_excl,:]=np.nan
        print('Data read')
        
        #define coordinates
        time=np.datetime64('1970-01-01T00:00:00')+tnum*1000*np.timedelta64(1,'ms')  
        range_id=np.arange(num_gates)
        
        #save as xarray
        data=xr.Dataset()
        data['rws']=xr.DataArray(data=rws,coords={'range':range_id,'time':time})
        data['snr']=xr.DataArray(data=snr,coords={'range':range_id,'time':time})
        
        #interpolate time on uniform linear space
        PF=np.polyfit(np.arange(len(tnum)), tnum, 1)
        tnum_uni=PF[1]+PF[0]*np.arange(len(tnum))
        tnum_uni=tnum_uni[(tnum_uni>tnum[0])*(tnum_uni<tnum[-1])]
        time_uni=tnum_uni*10**9*np.timedelta64(1,'ns')+np.datetime64('1970-01-01T00:00:00')
        data_uni=data.interp(time=time_uni) 
        
        #extract interpolated variables
        rws=np.array(data_uni['rws'])
        snr=np.array(data_uni['snr'])
        
        #stationarity check (based on stddev of binned stdev)
        bin_tnum_uni_nsi=np.arange(tnum_uni[0],tnum_uni[-1]+DT_nsi/2,DT_nsi)
        if len(bin_tnum_uni_nsi)<min_time_bins+1:
            bin_tnum_uni_nsi=np.linspace(tnum_uni[0],tnum_uni[-1],min_time_bins+1)
        
        for t1,t2 in zip(bin_tnum_uni_nsi[:-1],bin_tnum_uni_nsi[1:]):
            sel_t=(tnum_uni>=t1)*(tnum_uni<=t2)
            rws_std=vstack(rws_std,np.nanstd(rws[:,sel_t],axis=1))
            snr_std=vstack(snr_std,np.nanstd(snr[:,sel_t],axis=1))
            
        rws_nsi=np.nanstd(rws_std,axis=0)/np.nanmean(rws_std,axis=0)#non-stationarity index for rws
        snr_nsi=np.nanstd(snr_std,axis=0)/np.nanmean(snr_std,axis=0)#non-stationarity index for snr
        
        #exclude non-stationary gates
        rws_qc=rws.copy()
        rws_qc[rws_nsi>max_nsi,:]=np.nan
        rws_qc[snr_nsi>max_nsi,:]=np.nan
        
        #ACF estimation
        bin_tnum_uni=np.arange(tnum_uni[0],tnum_uni[-1]+DT/2,DT)
        for t1,t2 in zip(bin_tnum_uni[:-1],bin_tnum_uni[1:]):
            sel_t=(tnum_uni>=t1)*(tnum_uni<=t2)
            Nt=np.sum(sel_t)
            ACF=np.zeros((num_gates,N_lags))
            
            #detrending
            rws_avg=np.tile(np.nanmean(rws_qc[:,sel_t],axis=1),(Nt,1)).T
            rws_det=rws_qc[:,sel_t]-rws_avg
           
            #calculate ACF for each gate
            for i_r in range(num_gates):
                conv=np.correlate(rws_det[i_r,:],rws_det[i_r,:], mode='full')
                N=np.correlate(np.zeros(Nt)+1, np.zeros(Nt)+1, mode='full')
                ACF[i_r,:]=conv[Nt-1:Nt-1+N_lags]/N[Nt-1:Nt-1+N_lags]
                
                #check on variance vs 0-lag ACF
                if np.abs(ACF[i_r,0]-np.nanvar(rws_det[i_r,:]))>10**-10:
                    raise ValueError('Variance mismatch')
            
            #store all ACFs and median SNR
            ACF_all=vstack(ACF_all,ACF)
            snr_all=np.append(snr_all,np.nanmedian(snr[:,sel_t],axis=1))
        print(f'{m} completed')
#estimate noise (linear extrapolation method)
ACF_id=ACF_all.copy()
ACF_id[:,0]=(ACF_all[:,1]+(ACF_all[:,1]-ACF_all[:,2]))
noise=(ACF_all[:,0]-ACF_id[:,0])**0.5
noise[ACF_all[:,0]-ACF_id[:,0]<0]=np.nan
noise[noise<min_noise]=np.nan
failed=ACF_all[:,0]-ACF_id[:,0]<0

#noise curve vs snr
noise_avg=10**stats.binned_statistic(snr_all,np.log10(noise),statistic=lambda x:filt_mean(x),                       bins=bins_snr)[0]
noise_low=10**stats.binned_statistic(snr_all,np.log10(noise),statistic=lambda x:filt_BS_mean(x,p_value/2*100),      bins=bins_snr)[0]
noise_top=10**stats.binned_statistic(snr_all,np.log10(noise),statistic=lambda x:filt_BS_mean(x,(1-p_value/2)*100),  bins=bins_snr)[0]

#snr count
snr_count=np.histogram(snr_all,bins_snr)[0]

#failed
failed_avg=stats.binned_statistic(snr_all,failed,statistic='mean',bins=bins_snr)[0]

#qc
noise_avg_qc=noise_avg.copy()
noise_avg_qc[failed_avg>max_failed]=np.nan

#%% Output
Output=pd.DataFrame()
Output['SNR [dB]']=snr_avg
Output['Noise StDev [m/s]']=noise_avg_qc
Output.to_csv(os.path.join(cd,'data','snr_vs_noise.csv'))
    
#%% Plots
gs = gridspec.GridSpec(3,1,height_ratios=[2,0.5,0.5])
fig=plt.figure(figsize=(18,9))
ax=fig.add_subplot(gs[0])
plt.semilogy(snr_all,noise,'.k',alpha=0.1, markersize=10)
plt.errorbar(snr_avg,noise_avg,[noise_avg-noise_low,noise_top-noise_avg],color='r')
reals=~np.isnan(noise_avg)
plt.plot(snr_avg[reals],noise_avg[reals],'o-r',markersize=10,fillstyle='none')
plt.plot(snr_avg[reals],noise_avg_qc[reals],'.-r',markersize=15)
plt.grid()
plt.title('Noise curve based on '+str(len(ACF_all[:,0]))+' time series')
plt.ylabel('Measured noise st.dev. [m s$^{-1}$]')
plt.xlim([-30,0])
plt.xticks(np.arange(-30,1,5))
ax.set_xticklabels([])

ax=fig.add_subplot(gs[1])
ax.fill_between(snr_avg,snr_count*0,snr_count,color='b',alpha=0.5)
plt.ylabel('Occurence')
plt.xlim([-30,0])
plt.xticks(np.arange(-30,1,5))
plt.grid()
ax.set_xticklabels([])

ax=fig.add_subplot(gs[2])
ax.fill_between(snr_avg,failed_avg*0,failed_avg*100,color='b',alpha=0.5)
plt.xlabel(r'SNR [dB]')
plt.ylabel('Noise-free [%]')
plt.xlim([-30,0])
plt.ylim([0,100])
plt.xticks(np.arange(-30,1,5))
plt.grid()



