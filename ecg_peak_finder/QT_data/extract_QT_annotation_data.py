import os
import pandas as pd
import argparse
import sys



def main(args):
	os.mkdir('annotation_files')
	for file_name in args.file:
		if file_name[-3:] == 'atr':
			sample =  file_name[len('QT_raw_data/'):-4]
			# rdsamp: read sample
			# rdann: read annotation
			# -r: file name
			# -v: include headers
			# -c: Produce output in CSV (for rdsamp ONLY)
			# more info at:
			# https://archive.physionet.org/physiotools/wag/rdann-1.htm
			# https://archive.physionet.org/physiotools/wag/rdsamp-1.htm
			os.system('rdann -v -r ' + 'QT_raw_data/' + sample + ' -a atr' + '>annotation_files/' + sample + '_annotationATR.csv')
			df = pd.read_csv('annotation_files/' + sample + '_annotationATR.csv', delim_whitespace = True, header=0, engine='python')
			df.to_csv('annotation_files/' + sample + '_annotationATR.csv')
			print(sample)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Extracts the annotations for specified file name.')
	parser.add_argument('file', nargs='+', help='enter file name')
	args = parser.parse_args()
	main(args)
