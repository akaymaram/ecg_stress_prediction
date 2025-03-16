import argparse
import sys
import os

def main(args):
    if args.outDir:
        os.makedirs(args.outDir[0],exist_ok=True)
    for file in args.file:
        print(file.name,end=' --> ')

        lines = file.readlines()
        if args.outDir:
            out_dir = args.outDir[0]
        else:
            out_dir = os.path.dirname(file.name)
        base_name = os.path.basename(file.name)
        base_name = base_name.replace('.csv','') + "_rr.csv"
        start = 1
        end = len(lines)
        if args.header:
            start = 2
        if len(lines) <= start: continue
        fields = lines[start-1].split(',')
        prev = int(fields[args.peekCol])
        rate = 1.0/args.sampleRate
        rrs = [prev*rate] # this is not a valid RR timestamp , it is the time to the first sample 
        for idx in range(start,end):
            if len(lines[idx].strip()) == 0: continue
            fields = lines[idx].split(',')
            val = int(fields[args.peekCol])
            interval = val-prev
            prev = val
            interval = float(interval)*rate
            rrs.append(interval)
        out_path = os.path.join(out_dir,base_name)
        print(out_path)
        with open(out_path,'w') as f:
            f.write('Peek_Timestamp,RR_Interval\n')
            prev_time = 0
            for rr in rrs:
                prev_time = prev_time+rr
                f.write(str(prev_time)+","+str(rr*1000)+'\n') # save RR interval as ms num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract HRV features from RR file')
    parser.add_argument('-outDir', nargs=1, metavar=('DIR'), help='output dir, default in input loaction')
    parser.add_argument('--sampleRate', type=float, default=15.5, metavar='N', help="Sampling rate of input file")
    parser.add_argument('--peekCol', type=int, default=0, metavar='N', help="Column of Peek data (0-N)")
    parser.add_argument('--header', action='store_false',  default=True, help='Input CSV does not have a header row')
    parser.add_argument('-file', nargs='+', type=argparse.FileType('r'), help='csv file with peek idxes')
    args = parser.parse_args()
    main(args)