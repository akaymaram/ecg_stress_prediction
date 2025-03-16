import sys
import os
import operator
import re
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json
import argparse
import hrvanalysis as hrva


def chunk_min_size(min_time, data_chunks):
    total_len = 0
    for idx in range(len(data_chunks)-1,-1,-1):
        chunk_duration = data_chunks[idx][-1]['time']-data_chunks[idx][0]['time'] 
        if  chunk_duration < min_time:
            print('idx ',idx, ' is ', chunk_duration,' wanted ', min_time,' deleting chunk')
            del data_chunks[idx]
    return data_chunks

def chunk_non_contiguous(gap, data_chunks):
    chunks = []
    for data in data_chunks:
        prev_time = data[0]['time']
        chunk = []
        for entry in data:
            if entry['time'] -  prev_time > gap:
                print(str(gap)+'+ sec gap in time in the same data', entry['time'] -  prev_time)
                chunks.append(chunk.copy())
                chunk.clear()
            chunk.append(entry)
            prev_time = entry['time']
        chunks.append(chunk.copy())
    return chunks


def chunk_data_to_size(seconds, all_data, do_overlap=False):
    chunks = []

    for data in all_data:
        chunk = []
        chunk_half = []

        start_time = data[0]['time']
        start_time_half = data[0]['time'] + seconds/2

        for idx in range(len(data)):
            entry = data[idx]
            curr_time = entry['time']
            if curr_time >= start_time+seconds :
                chunks.append(chunk.copy())
                chunk.clear()
                start_time = curr_time
            elif do_overlap and curr_time >= start_time_half+seconds :
                chunks.append(chunk_half.copy())
                chunk_half.clear()
                start_time_half = curr_time

            if do_overlap and curr_time >= start_time_half: chunk_half.append(entry)
            chunk.append(entry)
        

        if len(chunk) > 2 and len(chunk_half) > 2:
            if chunk[0]['time'] < chunk_half[0]['time']:
                chunks.append(chunk.copy())
                chunks.append(chunk_half.copy())
            else:
                chunks.append(chunk_half.copy())
                chunks.append(chunk.copy())
        elif len(chunk) > 2:
            chunks.append(chunk.copy())
        elif len(chunk_half) > 2:
            chunks.append(chunk_half.copy())

    return chunks

def output_header(outFile):
    outFile.write("tag,start_time,end_time,duration")
    outFile.write(",mean_nni,sdnn,sdsd,nni_50,pnni_50,nni_20,pnni_20,rmssd,median_nni,range_nni,cvsd,cvnni,mean_hr,max_hr,min_hr,std_hr")
    outFile.write(",lf,hf,lf_hf_ratio,lfnu,hfnu,total_power,vlf")
    outFile.write("\n")

def output_row(outFile, tag, s_time, e_time, seconds, time_domain, freq_domain):
        if tag is None: tag = "Unknown"

        outFile.write(tag.replace(',',' ').replace('  ',' ') + ",")
        outFile.write(str(s_time) + ",")
        outFile.write(str(e_time) + ",")
        outFile.write(str(e_time-s_time) + ",")
        outFile.write(str(time_domain['mean_nni']) + ",")
        outFile.write(str(time_domain['sdnn']) + ",")
        outFile.write(str(time_domain['sdsd']) + ",")
        outFile.write(str(time_domain['nni_50']) + ",")
        outFile.write(str(time_domain['pnni_50']) + ",")
        outFile.write(str(time_domain['nni_20']) + ",")
        outFile.write(str(time_domain['pnni_20']) + ",")
        outFile.write(str(time_domain['rmssd']) + ",")
        outFile.write(str(time_domain['median_nni']) + ",")
        outFile.write(str(time_domain['range_nni']) + ",")
        outFile.write(str(time_domain['cvsd']) + ",")
        outFile.write(str(time_domain['cvnni']) + ",")
        outFile.write(str(time_domain['mean_hr']) + ",")
        outFile.write(str(time_domain['max_hr']) + ",")
        outFile.write(str(time_domain['min_hr']) + ",")
        outFile.write(str(time_domain['std_hr']) + ",")
        outFile.write(str(freq_domain['lf']) + ",")
        outFile.write(str(freq_domain['hf']) + ",")
        outFile.write(str(freq_domain['lf_hf_ratio']) + ",")
        outFile.write(str(freq_domain['lfnu']) + ",")
        outFile.write(str(freq_domain['hfnu']) + ",")
        outFile.write(str(freq_domain['total_power']) + ",")
        outFile.write(str(freq_domain['vlf']))
        outFile.write("\n")

def read_data(file, has_header, time_col, rr_col):
    start = 0
    if has_header:
        start = 1
    lines = file.readlines()
    end = len(lines)
    data = []
    for idx in range(start,end):
        if len(lines[idx].strip()) == 0: continue
        fields = lines[idx].split(',')
        if len(fields) < 2 : continue # need at least two columns
        data.append({'time':float(fields[time_col]),'rr':round(float(fields[rr_col]))})
    return data

def get_rrs(data):
    rr = []
    for entry in data:
         rr.append(entry['rr'])
    return rr

def main(args):

    output_header(args.out)
    for file in args.files:
        print(file.name)
        data = read_data(file, args.no_header, 0, 1)  #TODO take time and RR column indexes from command line args

        tag = os.path.basename(file.name)
        tag = tag.split('_')[0]
        data_chunks = chunk_non_contiguous(3, [data]) # break data into chunk if there is a N+ scond gap
        data_chunks= chunk_data_to_size(args.chunk, data_chunks, args.overlap)
        data_chunks = chunk_min_size(args.chunk*0.75, data_chunks) # discard chuck if not at least 0.75 of desired length 
        
        for chunk in data_chunks:
            rrs = get_rrs(chunk)
            time_domain_features = hrva.extract_features.get_time_domain_features(rrs)
            frequency_domain_features = hrva.extract_features.get_frequency_domain_features(rrs)
            output_row(args.out, tag, chunk[0]['time'], chunk[-1]['time'], args.chunk, time_domain_features, frequency_domain_features )

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract HRV features from RR file')
    parser.add_argument('--out', nargs='?', type=argparse.FileType('w'), help='output file path, will append if exists',default=sys.stdout)
    parser.add_argument('--no_header', action='store_false',  default=True, help='Input CSV does not have a header row')
    parser.add_argument('--fix', action='store_true',  default=False, help='Try to fix bad HRV values, save both')
    parser.add_argument('--tag', help='Add this tag to all data that is not already tagged')
    parser.add_argument('-chunk', nargs=1, metavar='SECs', default=60, type=check_positive, help='Extract data on N second chunks')
    parser.add_argument('--overlap', action='store_false',  default=True, help='Extracted data is overlaped by 50% of time value')
    parser.add_argument('-files', nargs='+', type=argparse.FileType('r'), help='json file with hrv data')
    args = parser.parse_args()
    main(args)