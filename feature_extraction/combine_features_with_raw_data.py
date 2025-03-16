import pandas as pd
import numpy as np
import sys
import os
import argparse

def main(args):
	df = pd.read_csv('driver_hrv_gsr.csv')
	# print(df)
	# count = 0
	# driver_count = 0
	# while count<len(df.iloc[:,0])-1:
	# 	x = df.iloc[count,0]
	# 	y = df.iloc[count+1,0]
	# 	print(df.iloc[count,1], df.iloc[count,2])
	# 	if (x != y):
	# 		driver_count+=1
	# 		print()
	# 	else:

	# 	count+=1


	ECG_list = []
	EMG_list = []
	foot_GSR_list = []
	hand_GSR_list = []
	HR_list = []
	RESP_list = []
	marker_list = []
	list_names = ['ECG', 'EMG', 'foot_GSR', 'hand_GSR', 'HR', 'RESP', 'marker']
	total_set = {'Elapsed time(s)', 'hand GSR(mV)', 'HR(bpm)', 'RESP(mV)', 'foot GSR(mV)', 'marker(mV)', 'ECG(mV)', 'EMG(mV)'}
	final_list = []


	count = 0
	for file in args.file:
		raw_df = pd.read_table(file,sep=',')
		num_columns = len(raw_df.iloc[0,:])
		missing_columns = total_set - set(raw_df.columns)

		#print('raw_df num columns', len(raw_df.iloc[0,:]))
		time_stamps = raw_df.iloc[:,0].to_numpy()
		time_stamps = time_stamps.astype(np.float)
		#print('time_stamps')
		#print(time_stamps)
		
		while count<len(df.iloc[:,0]):
			start_time_stamp = round(df.iloc[count,1], 3)
			end_time_stamp = round(df.iloc[count,2], 3)
			#print('start_time_stamp', start_time_stamp)
			#print('end_time_stamp', end_time_stamp)

			start_row = np.searchsorted(time_stamps, start_time_stamp)
			end_row = np.searchsorted(time_stamps, end_time_stamp)
			#print('\t\t\t\t\t\tstart_row {}    end_row {}'.format(time_stamps[start_row], time_stamps[end_row]))
			column_index = 1
			while column_index < num_columns:
				total_sum = 0
				for x in raw_df.iloc[start_row:end_row+1,column_index]:# +1 for differece between time_stamps indexing and raw_df indexing. +1 for end index inlcusion
					total_sum += float(x)
				mean = total_sum/len(raw_df.iloc[start_row:end_row+1,column_index])
				column_label = raw_df.columns[column_index]
				if column_label == 'ECG(mV)': ECG_list.append(mean)
				elif column_label == 'marker(mV)': marker_list.append(mean)
				elif column_label == 'RESP(mV)': RESP_list.append(mean)
				elif column_label == 'foot GSR(mV)': foot_GSR_list.append(mean)
				elif column_label == 'hand GSR(mV)': hand_GSR_list.append(mean)
				elif column_label == 'HR(bpm)': HR_list.append(mean)
				elif column_label == 'EMG(mV)': EMG_list.append(mean)
				#print('mean {}: {:<50}'.format(column_label, mean))
				column_index+=1
			for x in missing_columns:
				if x == 'ECG(mV)': ECG_list.append(np.nan)
				elif x == 'marker(mV)': marker_list.append(np.nan)
				elif x == 'RESP(mV)': RESP_list.append(np.nan)
				elif x == 'foot GSR(mV)': foot_GSR_list.append(np.nan)
				elif x == 'hand GSR(mV)': hand_GSR_list.append(np.nan)
				elif x == 'HR(bpm)': HR_list.append(np.nan)
				elif x == 'EMG(mV)': EMG_list.append(np.nan)



			if count >= len(df.iloc[:,0])-1:
				count+=1
				print('done')
				break

			current_tag = df.iloc[count,0]
			next_tag = df.iloc[count+1,0]
			if current_tag != next_tag:
				
				num = 0
				# while num < len(ECG_list):
				# 	for x in combined_list:print(ECG_list[num],EMG_list[num],foot_GSR_list[num],hand_GSR_list[num],HR_list[num],RESP_list[num], marker_list[num])
				# 	num+=1
				#print('\n\n\n\n')
				print(next_tag)
				print()
				count+=1
				break
			count+=1
	combined_list = [ECG_list,EMG_list,foot_GSR_list,hand_GSR_list,HR_list,RESP_list, marker_list]
	counter = 0
	print(df)
	while counter < len(combined_list):
		df.insert(4+counter, list_names[counter], combined_list[counter]) 
		counter+=1
	print(df)
	df.to_csv('final_file.csv', index = False)





			




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process HR device data, inputfile is CSV with time and ECG')
    parser.add_argument('file', nargs='+', help='full path to file(s) for driver')

    args = parser.parse_args()
    main(args)