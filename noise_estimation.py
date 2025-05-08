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
import matplotlib

plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

warnings.filterwarnings('ignore')

#%% Inputs
source=os.path.join(cd,'data','*scn')

#data format
skip=6#lines to skip
num_gates=34#number of gates
i_rws=1#column with velocity
i_snr=2#column with intensity
i_time=3#column with time

repeat=300

#noise estimation
N_lags=100#number of lags of ACF
max_nsi=0.5#maximum ratio of standard deviation standard deviation to mean standard deviation (os non-stationarity index, NSI)
DT_nsi=600# time window to evalue NSI
DT=600#[m] averaging time
min_time_bins=5#number of time bins to check non-stationarity
p_value=0.05#p_value for bootstrap
bins_snr=np.arange(-30.5,10.6)#[dB] bins in snr

#qc
rmin=96#[m] blind zone of the lidar
rmax=3000#[m] max range 
rws_max=40#[m/s] maximum rws discarded prior to the ACF estimation
min_noise=10**-10 #[m/s] minimum noise level

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
files=glob.glob(source)

snr_avg=(bins_snr[1:]+bins_snr[:-1])/2

#zeroing
tnum=np.zeros(len(files))
rws=np.zeros((num_gates,len(files)))
snr=np.zeros((num_gates,len(files)))
rws_std=[]
snr_std=[]
ACF_all=[]
snr_all=[]


#%% Main

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
    

print('Generating artificial data')
rws=np.tile(rws, (1, repeat))
snr=np.tile(snr, (1, repeat))
tnum=tnum[0]+np.arange(repeat*len(tnum))*np.diff(tnum)[0]

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
    snr_all=np.append(snr_all,np.nanmean(snr[:,sel_t],axis=1))

#estimate noise (linear extrapolation method)
ACF_id=ACF_all.copy()
ACF_id[:,0]=(ACF_all[:,1]+(ACF_all[:,1]-ACF_all[:,2]))
noise=(ACF_all[:,0]-ACF_id[:,0])**0.5
noise[ACF_all[:,1]<ACF_all[:,2]]=np.nan
noise[noise<min_noise]=np.nan

#noise curve
noise_avg=10**stats.binned_statistic(snr_all,np.log10(noise),statistic=lambda x:filt_mean(x),                       bins=bins_snr)[0]
noise_low=10**stats.binned_statistic(snr_all,np.log10(noise),statistic=lambda x:filt_BS_mean(x,p_value/2*100),      bins=bins_snr)[0]
noise_top=10**stats.binned_statistic(snr_all,np.log10(noise),statistic=lambda x:filt_BS_mean(x,(1-p_value/2)*100),  bins=bins_snr)[0]

#%% Plots
fig=plt.figure(figsize=(18,9))
main_ax = fig.add_axes([0.1, 0.3, 0.6, 0.6]) 
plt.semilogy(snr_all,noise,'.k',alpha=0.05, markersize=10)
plt.errorbar(snr_avg,noise_avg,[noise_avg-noise_low,noise_top-noise_avg],color='r')
reals=~np.isnan(noise_avg)
plt.plot(snr_avg[reals],noise_avg[reals],'.-r',markersize=15)
plt.grid()
plt.title('Noise curve based on '+str(len(ACF_all[:,0]))+' periods')
plt.xlabel(r'SNR [dB]')
plt.ylabel('Measured noise st.dev. [m s$^{-1}$]')
plt.xlim([-30,-5])
plt.xticks(np.arange(-30,-9,5))
plt.ylim([0.01,30])
