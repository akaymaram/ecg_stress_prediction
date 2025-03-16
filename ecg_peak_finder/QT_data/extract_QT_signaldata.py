import os
import pandas as pd
import argparse
import sys

def main(args):
	for file_name in args.file:
		if file_name[-3:] == 'dat':
			print(file_name)
			sample =  file_name[len('QT_raw_data/'):-4]
			# rdsamp: read sample
			# rdann: read annotation
			# -r: file name
			# -v: include headers
			# -c: Produce output in CSV (for rdsamp ONLY)
			# more info at:
			# https://archive.physionet.org/physiotools/wag/rdann-1.htm
			# https://archive.physionet.org/physiotools/wag/rdsamp-1.htm
			os.system('rdsamp -v -r QT_raw_data/' + sample + ' -p -v -c >' + sample + '.csv')
			df = pd.read_csv(sample + '.csv', header=0)
			print(df)
			df = df.drop(index = 0)
			df = df.reset_index(drop = True)
			df.to_csv(sample + '.csv')
			print(sample)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Extracts the signal for specified file name.')
	parser.add_argument('file', nargs='+', help='enter file name')
	args = parser.parse_args()
	main(args)