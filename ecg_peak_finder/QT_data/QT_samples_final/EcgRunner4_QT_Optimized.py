import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from Ecg14_QT_Optimized import *

def main(args):
	num = 0
	for file in args.file:
		num+=1
		if not os.path.isfile(file):
			print("Warning: File does not exist:",file)
			return
		if len(args.min_file_name) > 0 and file < args.min_file_name: continue

		ecg = Ecg(file, args.rate, args.timeCol, args.ecgCol, not args.no_header)

		if args.smooth:
			ecg.do_smoothing()


		ecg.normalize()

		print('('+str(num)+')', ecg.driver_file_name, end =' ')
		if args.auto:
			ecg.auto_invert()

		if args.invert:
			print('something is wrong')
			ecg.invert()


		min_index_to_plot = args.index_to_plot
		all_peaks = []		
		all_peaks = ecg.find_peaks(print_verbosity = 0, min_index_to_plot = min_index_to_plot, how_many_previous_peaks_to_use = 10)
		indices = pd.Series(all_peaks)
		y_values = pd.Series(ecg.total_peak_raw_ecg_values)
		new_list = []
		x = 0
		while x < len(indices):
			new_list.append((indices[x], y_values[x]))
			x+=1


		df = pd.concat([indices, y_values], axis = 1)
		df = pd.DataFrame({'peak_index': indices, 'peak_raw_ecg_value': y_values})
		if not os.path.isdir('peak_indicesQ'):
			os.mkdir('peak_indicesQ')

		file_name = 'peak_indicesQ/' + ecg.driver_file_name[:-4] + '_peak_indices.csv'
		if os.path.isfile(file_name):
			old_df = pd.read_csv(file_name)
			x = 0
			old_list = []
			while x < len(old_df.iloc[:,0]):
				old_list.append((old_df.iloc[x,0], old_df.iloc[x,1]))
				x+=1

			old_set = set(old_list)
			new_set = set(new_list)
			old_minus_new = old_set - new_set
			new_minus_old = new_set - old_set
			if len(old_minus_new) == 0 and len(new_minus_old) == 0:
				print('they completely match.\n')
			else:
				old_minus_new_list = list(old_minus_new)
				old_minus_new_list.sort()
				print('old_minus_new:', old_minus_new_list)
				new_minus_old_list = list(new_minus_old)
				new_minus_old_list.sort()
				print('new_minus_old:', new_minus_old_list)
				for x in old_minus_new_list: ecg.plot_index([x[0]])
				for x in new_minus_old_list: ecg.plot_index([x[0]])
				x = ecg_input(['', ' '], 'save new or keep old?(ENTER for new, SPACE ENTER for old)')
				if x == '':
					plt.close('all')
					df.to_csv(file_name, index = False)
					print('new file saved.\n')
				elif x == ' ':
					plt.close('all')
					print('kept old file.\n')
		else:
			df.to_csv(file_name, index = False)
			print('no previous file found; saved this file.\n')
		continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process HR device data, inputfile is CSV with time and ECG')
    parser.add_argument('--rate',  type=int, default=250, help='Sampling rate of input file (default: 16)')
    parser.add_argument('--no_header', action='store_true',  default=False, help='CSV file does not have a header row')
    parser.add_argument('--invert', action='store_true',  default=False, help='Invert Values')
    parser.add_argument('--auto',   action='store_true',  default=True, help='Try to determine if Invert is needed')
    parser.add_argument('--smooth',   action='store_true',  default=True, help='Run savgol_filter  on ecg data')
    parser.add_argument('--timeCol',  type=int, default=1, help='Column in CSV file that has Time (default: 0)')
    parser.add_argument('--ecgCol',  type=int, default=2, help='Column in CSV file that has ECG (default: 1)')
    parser.add_argument('file', nargs='+', help='full path to file(s) for driver')
    parser.add_argument('--index_to_plot', type=int, default=-1, help='min index to plot')
    parser.add_argument('--min_file_name', default='', help='min file to process')
    args = parser.parse_args()
    main(args)
