import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit as nk
import pandas as pd
import seaborn as sns
import scipy as sp
import statistics 
from enum import Enum
from scipy.signal import savgol_filter

def ave_zeros(data):
    new_data = data[:]
    l = len(new_data)

    if l < 3 : return data

    # get reid of leading zeros
    first_good = 0
    for idx in range(0,l-1):
        if new_data[idx] != 0: 
            break
        first_good += 1
    if first_good > 0:
        for idx in range(0,first_good):
            new_data[idx] = new_data[first_good]

    for idx in range(1,l-2):
        if new_data[idx] == 0:
            new_data[idx] = (new_data[idx-1]+new_data[idx+1])/2.0
    return new_data

def remove_spikes(data):
    new_data = data[:]
    l = len(new_data)

    if l < 3 : return data

    for idx in range(50,l):
        if new_data[idx]-new_data[idx-1] >1 or new_data[idx]-new_data[idx-1] < -1:
            new_data[idx] = new_data[idx-1]
    return new_data

def read_drive_file(f_name):
    data = []
    times = []
    lines = open(f_name,"r").readlines()
    if len(lines) < 2 :
        return data
    rate = 15.5
    if 'drive03' in f_name:
        rate = 31
    for line in lines[1:]:
        if len(line.strip(', \n\r')) == 0: continue
        fileds = line.strip().split(',')
        if len(fileds) < 2: continue
        time_stamp = float(fileds[0])
        val = float(fileds[3])
        times.append(time_stamp)
        data.append(val)
    return times,data,rate

def get_gsr_features(gsr_name):
    _, data, rate = read_drive_file(gsr_name)
    data1 = ave_zeros(data)
    data2 = remove_spikes(data1)
    mi = min(data2) - 0.001
    data3 = [x - mi for x in data2] # set min near zero
    

    np_data = np.array(data3, dtype='float64')
    bio = nk.bio_process( eda=np_data,  sampling_rate=rate)
    df_results = nk.z_score(bio["df"])
    df_results = df_results.drop(columns=['EDA_Raw', 'EDA_Filtered','SCR_Onsets', 'SCR_Recoveries']) # leaves: EDA_Phasic EDA_Tonic SCR_Peaks

    return df_results,rate

def get_gsr_features_pysiology(gsr_name):
    _, data, rate = read_drive_file(gsr_name)
    data1 = ave_zeros(data)
    data2 = remove_spikes(data1)
    mi = min(data2) - 0.001
    data3 = [x - mi for x in data2] # set min near zero

    res = pysiology.electrodermalactivity.analyzeGSR(data3, int(rate))
    pt = pysiology.electrodermalactivity.getPhasicAndTonic(data3, int(rate))
    peaks = [np.nan] * len(pt[0])
    for ii in range(len(res)):
        idx = res[ii]['peak']['peakMax']
        peaks[idx] = pt[0][idx]
    all_data = {'EDA_Phasic':pt[1], 'EDA_Tonic':pt[0], 'SCR_Peaks':peaks}
    df = pd.DataFrame.from_dict(all_data)

    return df,rate

if len(sys.argv) < 3: 
    print("Usage: driver_hrv.csv driver_data_final/drive*.csv")
    sys.exit(1)

all_hrv_df = pd.read_csv(sys.argv[1])

all_gsr = {'tonic_mean':[],'tonic_var':[],'phasic_mean':[],'phasic_var':[],'phasic_sum':[],'phasic_peaks':[],'phasic_sum_over_1':[],'phasic_peaks_over_1':[]}
for f in sys.argv[2:]:
    print(f)
    tag = os.path.basename(f).replace(".csv", "")
    gsr_df,rate = get_gsr_features(f)
    rate = 1/rate

    all_peaks = gsr_df['SCR_Peaks']
    s = np.std(~np.isnan(all_peaks))
    m = np.mean(~np.isnan(all_peaks))
    c = np.count_nonzero(~np.isnan(all_peaks))
    print(c,m,s)

    hrv_df = all_hrv_df.loc[all_hrv_df['tag'] == tag]
    for idx, row in hrv_df.iterrows():
        start = int(row['start_time']/rate)
        end = int(row['end_time']/rate)
        t_segment = np.array(gsr_df['EDA_Tonic'][start:end+1])
        t_peaks = sp.signal.find_peaks(t_segment)
        p_segment = np.array(gsr_df['EDA_Phasic'][start:end+1])
        p_peaks = gsr_df['SCR_Peaks'][start:end+1]
        p_greater_than_1 = p_peaks[p_peaks > m]


        #print(start, end, end-start)
        all_gsr['tonic_mean'].append(t_segment.mean())
        all_gsr['tonic_var'].append(np.var(t_segment))
        all_gsr['phasic_mean'].append(p_segment.mean())
        all_gsr['phasic_var'].append(np.var(p_segment))
        all_gsr['phasic_sum'].append(p_peaks.sum())
        all_gsr['phasic_peaks'].append(int( np.count_nonzero(~np.isnan(p_peaks))))
        all_gsr['phasic_sum_over_1'].append(p_greater_than_1.sum())
        all_gsr['phasic_peaks_over_1'].append(int(len(p_greater_than_1)))

all_grs_df = pd.DataFrame.from_dict(all_gsr)
all_df = pd.concat([all_hrv_df, all_grs_df], axis=1, sort=False)

all_df.to_csv("driver_hrv_gsr.csv",index=False, float_format='%.6f')