import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy import signal
from scipy.signal import savgol_filter
import find_peak_window_2
import statistics
import math


# TO DO:
# optimize run time
# simplify and match the calibration() filtering with find_peaks() filtering
# bad segment detector
# savgol for whole or just segments
# get rid of outliers before nromalization

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx]

def ecg_input(list_of_allowed_inputs = [''], message = None , end = '\n'):
	bad_input = True
	while bad_input == True:
		print(message, end = end)
		x = input()
		if x in list_of_allowed_inputs:
			bad_input = False
			return x
		else:
			print('bad input. try again')

def plot_horizontal_line(list_of_y_values = [], c = 'r'):
	for x in list_of_y_values: plt.axhline(x, c = c)


def plot_vertical_line(list_of_x_values = [], c = 'r'):
	for x in list_of_x_values: plt.axvline(x, c = c)

def plot_dots(x_values = [], y_values = [], c = 'r'):
	if len(x_values) != len(y_values):
		print('bad input lists. list lengths do not match.')
		return None
	index = 0
	while index < len(x_values):
		plt.plot(x_values[index], y_values[index], '.', c = c)
		index+=1






class Ecg:
	def __init__(self, file_name, sampling_rate = 250, timeCol = 1, ecgCol = 2, has_header = True):
		self.sampling_rate = sampling_rate
		self.total_peak_indices = []
		self.total_peak_ecg_values = []
		self.total_peak_raw_ecg_values = []
		self.total_peak_prominences = []
		self.total_peak_timestamps = []
		self.total_RR_intervals = []
		self.initial_peak_timestamps = []
		self.initial_RR_intervals = []
		self.possible_indices_of_peak_anamolies = []
		self.driver_file_name = file_name

		if has_header:
			dataframe = pd.read_csv(self.driver_file_name)
		else:
			dataframe = pd.read_csv(self.driver_file_name, header=None)
		self.raw_ecg_data = dataframe.iloc[: ,ecgCol]
		self.ecg_data = self.raw_ecg_data
		self.time_data = np.array(dataframe.iloc[: ,timeCol])




	def normalize(self): self.ecg_data = (self.ecg_data - self.ecg_data.min())/(self.ecg_data.max() - self.ecg_data.min())

	def do_smoothing(self):
		window = self.sampling_rate
		if self.sampling_rate % 2 == 0:
			window += 1

		self.ecg_data = self.ecg_data - savgol_filter(self.ecg_data, window, 3)

	def invert(self):
		self.ecg_data = (self.ecg_data * -1)+1
		print('inverted')


	def auto_invert(self, num_chunks = 100, num_sampling_rate_each_chunk = 10):
		score = 0
		for e in range(0, num_chunks*self.sampling_rate, self.sampling_rate*num_sampling_rate_each_chunk):
			mean = statistics.mean(self.ecg_data[e : e+self.sampling_rate*num_sampling_rate_each_chunk])
			std = statistics.stdev(self.ecg_data[e : e+self.sampling_rate*num_sampling_rate_each_chunk])
			not_enough_points = True
			pos = []
			neg = []
			num_stds = 3
			while(not_enough_points):
				pos = []
				neg = []
				for v in self.ecg_data[e : e+self.sampling_rate*num_sampling_rate_each_chunk]:

					if v < mean-num_stds*std: neg.append(abs(v-mean))
					elif v > mean+num_stds*std: pos.append(abs(v-mean))


				#plot auto_invert
				# plt.plot(self.ecg_data[e : e+self.sampling_rate*num_sampling_rate_each_chunk], 'o')
				# plt.axhline(mean, c = 'r')
				# plt.axhline(mean+num_stds*std, c = 'r')
				# plt.axhline(mean-num_stds*std, c = 'r')
				# plt.axhline(mean, c = 'y')
				# plt.axhline(statistics.median(self.ecg_data[e : e+self.sampling_rate*num_sampling_rate_each_chunk]), c = 'g')

				#plt.show()



				if len(neg)+ len(pos) > num_sampling_rate_each_chunk: not_enough_points = False
				else: num_stds -= .5
			

			if len(neg) > len(pos):
				score -= 1
			else:
				score += 1

		if  len(neg) > len(pos):
			self.invert()
		else: print()

		# just another idea
		# segment = self.ecg_data[: 6*self.sampling_rate]
		# if  max(segment) - statistics.median(segment) < statistics.median(segment) - min(segment):
		# 	print('top dis:', max(segment) - statistics.median(segment))
		# 	print('bottom dis:', statistics.median(segment) - min(segment))
		# 	self.invert()


	def calibrate(self, start_index = 0, input_verbosity = 1, plot_list_of_peaks_one_by_one = False):
		if start_index == 0:
			start_index = int(self.sampling_rate/2)
		num_secconds_in_calibration = 6
		end_index = start_index + self.sampling_rate*num_secconds_in_calibration
		max_heart_rate_bpm = 200
		max_number_of_peaks = int((max_heart_rate_bpm/60)*num_secconds_in_calibration)
		min_distance = int(self.sampling_rate/3)
		height = statistics.mean(self.ecg_data[start_index:end_index])/2
		prominence = 0
		count = 0
		segment_peaks = None
		too_many_peaks = True
		while too_many_peaks:
			if (count % 3) == 2:
				min_distance = min_distance+math.ceil(self.sampling_rate/25)
			elif (count % 3) == 1:
				prominence = prominence+.02
			elif (count % 3) == 0:
				height = height+.02
			segment_peaks, _ = signal.find_peaks(self.ecg_data[start_index: end_index], distance = min_distance, height = height, prominence = prominence)
			if len(segment_peaks) <= max_number_of_peaks: too_many_peaks = False
			else: count+=1


		list_of_tuple_of_peaks = []
		prom,_,_ = signal.peak_prominences(self.ecg_data[start_index:end_index], segment_peaks)
		prominence = max(prom)/4

		count = 0
		while len(segment_peaks) > 3 and count < 30:
			if (count % 3) == 3:
				min_distance = min_distance+math.ceil(self.sampling_rate/25)
			elif (count % 3) == 1:
				prominence = prominence+.02
			elif (count % 3) == 0:
				prominence = prominence+.02
			segment_peaks, _ = signal.find_peaks(self.ecg_data[start_index: end_index], distance = min_distance, height = height, prominence = prominence)
			segment_peaks+=start_index
			list_of_tuple_of_peaks.append(tuple(segment_peaks))
			count+=1

		no_repeat_list_of_tuple_of_peaks = list(set(list_of_tuple_of_peaks))
		no_repeat_list_of_tuple_of_peaks.sort()



		if plot_list_of_peaks_one_by_one == True:
			for x in no_repeat_list_of_tuple_of_peaks:
				x = list(x)
				plot_horizontal_line(list_of_y_values = [.68*max(self.ecg_data[start_index:end_index])], c = 'r')
				self.plot_index(x, start_index = start_index, end_index = end_index)
	


	
		index = 0
		while index <  len(no_repeat_list_of_tuple_of_peaks):
			if len(no_repeat_list_of_tuple_of_peaks[index]) < 3 or len(no_repeat_list_of_tuple_of_peaks[index]) > 12:
				del no_repeat_list_of_tuple_of_peaks[index]
			else:
				index+=1



		good_height_peaks = []
		for tupl in no_repeat_list_of_tuple_of_peaks:
			for peak in tupl:
				bad_height = False
				if self.ecg_data[peak] < .68*max(self.ecg_data[start_index:end_index]):
					bad_height = True
					break
			if bad_height == True: continue
			good_height_peaks.append(tupl)



		
		no_repeat_array_peaks = np.array(good_height_peaks)






		#step-by-step testing
		# if plot_list_of_peaks_one_by_one == True:
		# 	for tup in no_repeat_array_peaks:
		# 		plt.plot(self.ecg_data[start_index: end_index])
		# 		plt.plot(list(tup), self.ecg_data[list(tup)], 'x')
		# 		plt.legend([self.driver_file_name])
		# 		plt.show(block = False)
		# 		user_input = ecg_input(message = 'press ENTER for next.')
		# 		if user_input == '':
		# 			plt.close()




		good_RR_interval_peaks = []
		good_RR_interval_AND_prominence_peaks = []
		index = 0
		nothing_found = False
		message = ''
		while index < len(no_repeat_array_peaks):
			peak_timestamps = self.time_data[np.array(no_repeat_array_peaks[index])]
			RR_intervals = np.diff(peak_timestamps)
			RR_intervals_std = statistics.stdev(RR_intervals)
			if RR_intervals_std < .26: good_RR_interval_peaks.append(index)
			index+=1




		if len(good_RR_interval_peaks) > 0:
			if plot_list_of_peaks_one_by_one == True:
				for x in good_RR_interval_peaks:
					plt.plot(self.ecg_data[start_index: end_index])
					the_list = list(no_repeat_array_peaks[x]) #  from tuple to list
					plt.plot(the_list, self.ecg_data[the_list], 'x')
					plt.legend([self.driver_file_name+' good_RR_interval_peaks'])
					plt.show(block = False)
					user_input = ecg_input(message = 'press ENTER for next.')
					if user_input == '':
						plt.close()

			for index in good_RR_interval_peaks:
				prom,_,_ = signal.peak_prominences(self.ecg_data[: end_index+int(self.sampling_rate/2)], no_repeat_array_peaks[index])
				good_prom_condition = True
				for x in prom:
					if x < .31*max(prom):
						good_prom_condition = False
				if good_prom_condition:
					good_RR_interval_AND_prominence_peaks.append(index)
			if len(good_RR_interval_AND_prominence_peaks)<1:
				nothing_found = True
				message = 'Bad peak prominences. There is a peak that has a prominence less than .4*maximum peak prominence.'

		else:
			nothing_found = True
			message = 'Bad rr_intervals. The standard deviation of the rr_intervals is beyond .26'

		if nothing_found == True:
			print(message)
			if input_verbosity > 0:
				plt.plot(self.ecg_data[start_index: end_index])
				plt.plot(segment_peaks, self.ecg_data[segment_peaks], 'x')
				plt.legend([self.driver_file_name, message])
				plt.show(block = False)
				x = ecg_input(message = 'Press ENTER to skip half a sampling rate.')
				if x == '':
					plt.close('all')
					self.calibrate(start_index = start_index+int(self.sampling_rate/2), input_verbosity = input_verbosity)
					return 1
			else:
				plt.close('all')
				self.calibrate(start_index = start_index+int(self.sampling_rate/2), input_verbosity = input_verbosity)
				return 1




		#fft
		fft_period, fft_y = find_peak_window_2.find_period_fft(self.ecg_data[start_index: end_index], math.ceil(self.sampling_rate/3), math.ceil(1.5*self.sampling_rate), self.sampling_rate)

		max_list_size_index = good_RR_interval_AND_prominence_peaks[0]
		max_list_size_length = 0
		for index in good_RR_interval_AND_prominence_peaks:
			index_difference = np.diff(no_repeat_array_peaks[index])
			mean = np.mean(index_difference)
			if  mean < fft_period*1.2 and mean > fft_period*0.60:
				if len(no_repeat_array_peaks[index]) > max_list_size_length:
					max_list_size_index = index
					max_list_size_length = len(no_repeat_array_peaks[index])


		segment_peaks = list(no_repeat_array_peaks[max_list_size_index]) # tuple to list

		initial_peak_timestamps = np.array(self.time_data[segment_peaks])
		initial_RR_intervals = np.diff(initial_peak_timestamps)

		x = ''
		if input_verbosity > 0:
			plt.plot(self.ecg_data[start_index: end_index])
			plt.plot(segment_peaks, self.ecg_data[segment_peaks], 'o')
			plt.legend([str(self.driver_file_name)+' calibration mode'])
			plt.show(block=False)
			x = ecg_input(['', 's'], message = 'press ENTER for good peaks and S to skip half a sampling rate.', end = '')

		if x == '':
			prom,L,R = signal.peak_prominences(self.ecg_data[: end_index], segment_peaks)
			self.total_peak_indices.extend(segment_peaks)
			self.total_peak_raw_ecg_values.extend(self.raw_ecg_data[segment_peaks])
			self.total_peak_ecg_values.extend(self.ecg_data[segment_peaks])
			self.total_peak_prominences.extend(prom)
			self.total_peak_timestamps.extend(initial_peak_timestamps)
			self.total_RR_intervals.extend(initial_RR_intervals)
			plt.close('all')
			return 1
		elif x == 's':
			plt.close('all')
			self.calibrate(start_index = start_index+int(self.sampling_rate/2), input_verbosity = input_verbosity)
			return 1


	def calib(self, min_distance = 1, start_index = 0, height = 0, prominence = 0):
		segment_peaks, _ = signal.find_peaks(self.ecg_data[start_index: end_index], distance = min_distance, height = height, prominence = prominence)

		initial_peak_timestamps = np.array(self.time_data[segment_peaks])
		initial_RR_intervals = np.diff(initial_peak_timestamps)

		
		fft_period, fft_y = find_peak_window_2.find_period_fft(self.ecg_data[0: 1000], math.ceil(self.sampling_rate), math.ceil(1.5*self.sampling_rate), self.sampling_rate)
		plt.plot(fft_period)

		plt.show()
			
	

	def plot_time(self, start_index, end_index):
		plt.plot(self.time_data[start_index : end_index], self.ecg_data[start_index : end_index], label = 'Time instead of index')
		peaks = []
		for peak in self.total_peak_indices:
			if peak > start_index and peak < end_index:
				peaks.append(peak)
		plt.plot(self.time_data[peaks], self.ecg_data[peaks], 'x')
		plt.show()


	def plot_index(self, list_of_points = [], list_of_labels= [], start_index = -1, end_index = -1):

		if start_index < 0: start_index = max(min(list_of_points) -self.sampling_rate, 0)
		if end_index < 0: end_index = max(list_of_points) +self.sampling_rate
		plt.plot(self.ecg_data[start_index: end_index])
		x = 0
		if len(list_of_labels) > 0:
			while x < len(list_of_points):
				point = list_of_points[x]
				plt.plot(point, self.ecg_data[point], 'x', label = list_of_labels[x], color = 'r')
				x+=1
			plt.legend()

		else:
			while x < len(list_of_points):
				point = list_of_points[x]
				plt.plot(point, self.ecg_data[point], 'x', color = 'r')
				x+=1
	
		plt.show(block = False)

		x = ecg_input(message = 'press ENTER for next.')
		if x == '':
			plt.close('all')


	def plot_filtering(self, list_of_points = [], list_of_vertical_lines = [], end_index = 0, min_rr_interval = 0, max_rr_interval= 0, list_of_horizontal_lines= [], min_prom=0, plot_verbosity = 1):

		if plot_verbosity < 1:
			if len(list_of_points) == 1: return 0 #don't plot
		start = -9
		if len(self.total_peak_indices) < 10: start = 0


		

		if len(list_of_vertical_lines) > 0: plot_vertical_line(list_of_vertical_lines)
		else:

			last_peak_index = self.total_peak_indices[-1]
			last_peak_timestamp = self.total_peak_timestamps[-1]
			end_plot_index = max(last_peak_index+self.sampling_rate*6, end_index)

			#min_x
			closest_min_timestamp = find_nearest(self.time_data[last_peak_index:end_plot_index], last_peak_timestamp+min_rr_interval)
			min_index = np.where(self.time_data[last_peak_index:end_plot_index] == closest_min_timestamp)
			min_index += last_peak_index
			start_index = min_index[0][0]
			start_index = last_peak_index + min_rr_interval*self.sampling_rate
			plt.axvline(start_index, c = 'r')

			#max_x
			closest_max_timestamp = find_nearest(self.time_data[last_peak_index:end_plot_index], last_peak_timestamp+max_rr_interval)
			max_index = np.where(self.time_data[last_peak_index:end_plot_index] == closest_max_timestamp)
			max_index += last_peak_index
			end_index = max_index[0][0]
			end_index = last_peak_index + max_rr_interval*self.sampling_rate
			plt.axvline(end_index, c = 'r')


		#ecg, previous peaks (o), possible peaks (x)
		plt.plot(self.ecg_data[self.total_peak_indices[start]: end_plot_index])
		plt.plot(self.total_peak_indices[start:], self.ecg_data[self.total_peak_indices[start:]], 'o')
		plt.plot(list_of_points, self.ecg_data[list_of_points], 'x')


		#min_y
		for y in list_of_horizontal_lines: plot_horizontal_line(list_of_horizontal_lines)

		#prom dots
		prom,L,R = signal.peak_prominences(self.ecg_data[last_peak_index: end_plot_index], list_of_points-last_peak_index)
		base_line = self.ecg_data[list_of_points].tolist() - prom
		plot_dots(list_of_points, [y-min_prom for y in self.ecg_data[list_of_points]])
		plot_dots(list_of_points, base_line, 'g')



		plt.show(block=False)

		x = ecg_input(message = 'press ENTER for next.')
		if x == '':
			plt.close('all')







	# verbosity: how much to print
	def find_peaks(self,  how_many_previous_peaks_to_use = 1, input_verbosity = 1, min_index_to_plot = -1, print_verbosity = 0, plot_verbosity = 1):

		if len(self.total_peak_indices) == 0:
			self.calibrate(input_verbosity = input_verbosity)
		print('working on it...')

		num_peak_to_use = min(max(0, len(self.total_peak_indices)), how_many_previous_peaks_to_use)*(-1)


		if min_index_to_plot < 0: min_index_to_plot = len(self.ecg_data)-1

		end_index = self.total_peak_indices[-1]
		extend_range = 1
		while (end_index < len(self.ecg_data)) and extend_range == 1:
			start_index = end_index
			while end_index < start_index+self.sampling_rate*6:
				end_index+=1
				if end_index == len(self.ecg_data)-1:
					break	
			if print_verbosity > 0 or start_index > min_index_to_plot:
				print('start:', start_index)

			good_RR_interval_peaks = []

			# step-by-step illustration
			# if start_index > 1100:
			# 	start = -9
			# 	if len(self.total_peak_indices) < 10:
			# 		start = 0
			# 	value = self.total_peak_timestamps[-1]+self.total_RR_intervals[-1]*1.5
			# 	close_val = find_nearest(np.array(self.time_data), value)
			# 	ind = np.where(np.array(self.time_data) == close_val)
			# 	plt.plot(self.ecg_data[self.total_peak_indices[start]+ start:segment_peaks[-1]+10])
			# 	plt.plot(self.total_peak_indices[start:], self.ecg_data[self.total_peak_indices[start:]], 'o')
			# 	plt.axvline(ind[0][0], c = 'r', ymin=0.1, ymax=0.9)
			# 	plt.plot(segment_peaks, self.ecg_data[segment_peaks], 'x')
			# 	plt.show()



			x_std = statistics.stdev(self.total_RR_intervals[num_peak_to_use:])
			x_median = statistics.median(self.total_RR_intervals[num_peak_to_use:])

			p_std = statistics.stdev(self.total_peak_prominences[num_peak_to_use:])
			p_median = statistics.median(self.total_peak_prominences[num_peak_to_use:])

			h_std = statistics.stdev(self.total_peak_ecg_values[num_peak_to_use:])
			h_median = statistics.median(self.total_peak_ecg_values[num_peak_to_use:])
			h_mean = statistics.mean(self.total_peak_ecg_values[num_peak_to_use:])
			# if h_mean > h_median: h_median = h_mean

			segment_peaks, _ = signal.find_peaks(self.ecg_data[start_index: end_index], distance = int(self.sampling_rate/3), prominence = p_median/4)
			segment_peaks += start_index

			# plot all raw caught peaks (before filtering)
			# if start_index > min_index_to_plot: self.plot_index(segment_peaks)

			the_peak = None
			found_peak = False
			ultra_extend = False
			extend_x = 0
			extend_y = 0
			extend_prom =0
			tries = 29

			if(len(segment_peaks) == 0):
				if end_index != len(self.ecg_data)-1:
					print('no peaks found at all')
					extend_range = 3
					end_index = self.total_peak_indices[-1]
				else:
					break
			else:
				while ((extend_x+extend_y+extend_prom) < tries or extend_x < 60) and found_peak == False:
					if print_verbosity > 0 or start_index > min_index_to_plot:
						print('tries:' ,extend_x+extend_y+extend_prom)
						print('extend_x:', extend_x)
					min_rr_interval = max(min(x_median-(3*x_std*(extend_x+1)), x_median*(0.45-.03*extend_x)), x_median*0.2)
					max_rr_interval = max(x_median+(3.5*x_std*(extend_x+1)), x_median*(1.2+.3*extend_x))
					min_y = max(min(h_median-(0*h_std+1.5*h_std*(extend_y)), h_median*(.8-.2*extend_y)), h_median*.3)
					min_prom = max(min(p_median-(.9*p_std+p_std*(extend_prom)), p_median*(.69-.1*extend_prom)), p_median*.25)

					# print('h_median*(.8:', h_median*(.8))
					for x in segment_peaks:
						if x not in good_RR_interval_peaks:
							possible_RR_interval = self.time_data[x] - self.total_peak_timestamps[-1]
							if possible_RR_interval > min_rr_interval and possible_RR_interval < max_rr_interval:
									good_RR_interval_peaks.append(x)

					if print_verbosity > 0 or start_index > min_index_to_plot:
						print('good_RR_interval_peaks:', good_RR_interval_peaks)	
					if len(good_RR_interval_peaks)>0:
						good_RR_interval_AND_height_peaks = []
						for x in good_RR_interval_peaks:
							if self.ecg_data[x] > min_y:
								if x not in good_RR_interval_AND_height_peaks:
									good_RR_interval_AND_height_peaks.append(x)
						if print_verbosity > 0 or start_index > min_index_to_plot:
							print('good_RR_interval_AND_height peaks:', good_RR_interval_AND_height_peaks)	
						if len(good_RR_interval_AND_height_peaks) > 0:
							good_RR_interval_AND_height_AND_prominence_peaks = []
							prom,L,R = signal.peak_prominences(self.ecg_data[: end_index], good_RR_interval_AND_height_peaks)
							
							index = 0
							while index < len(prom):
				
								if prom[index] > min_prom:
									the_peak_index = good_RR_interval_AND_height_peaks[index]
									if the_peak_index not in good_RR_interval_AND_height_AND_prominence_peaks:
										good_RR_interval_AND_height_AND_prominence_peaks.append(the_peak_index)
								index+=1
							if print_verbosity > 0 or start_index > min_index_to_plot:
								print('good_RR_interval_AND_height_AND_prominence_peaks:', good_RR_interval_AND_height_AND_prominence_peaks)
							if len(good_RR_interval_AND_height_AND_prominence_peaks)>0:
								prom,L,R = signal.peak_prominences(self.ecg_data[: end_index], good_RR_interval_AND_height_AND_prominence_peaks)


								#pick highest prom peak
								# max_prom = 0
								# for x in prom:
								# 	if x > max_prom: max_prom = x
								# inde = np.where(prom == max_prom)[0][0]
								# the_peak_index = good_RR_interval_AND_height_AND_prominence_peaks[inde]

								# if len(good_RR_interval_AND_height_AND_prominence_peaks)>4:
								# 	print('too many peaks')
								# 	print('Caliberate')
								# 	self.plot_filtering(good_RR_interval_AND_height_AND_prominence_peaks, [], end_index, min_rr_interval, max_rr_interval, [min_y], min_prom, plot_verbosity)
								# 	self.calibrate(start_index = end_index, input_verbosity = input_verbosity)
								# 	return self.find_peaks(how_many_previous_peaks_to_use = how_many_previous_peaks_to_use, input_verbosity = input_verbosity, min_index_to_plot = min_index_to_plot, print_verbosity = print_verbosity, plot_verbosity = plot_verbosity)


								#pick first peak
								max_prom = prom[0]
								the_peak_index = min(good_RR_interval_AND_height_AND_prominence_peaks)



								if start_index > min_index_to_plot:
									self.plot_filtering(good_RR_interval_AND_height_AND_prominence_peaks, [], end_index, min_rr_interval, max_rr_interval, [min_y], min_prom, plot_verbosity)





								self.total_peak_prominences.append(max_prom)
								self.total_peak_indices.append(the_peak_index)
								self.total_peak_raw_ecg_values.append(self.raw_ecg_data[the_peak_index])
								self.total_peak_ecg_values.append(self.ecg_data[the_peak_index])
								self.total_peak_timestamps.append(self.time_data[the_peak_index])
								self.total_RR_intervals.append(self.total_peak_timestamps[-1]-self.total_peak_timestamps[-2])
								found_peak = True



							else:
								if start_index > min_index_to_plot:
									self.plot_filtering(good_RR_interval_AND_height_peaks, [], end_index, min_rr_interval, max_rr_interval, [min_y], min_prom, plot_verbosity)
								if (extend_x+extend_y+extend_prom) >= tries-2 or ultra_extend == True:
									ultra_extend = True
									extend_y = 0
									extend_prom = 0									
									extend_x+=4
									if extend_x > 60:
										if end_index == len(self.ecg_data)-1:
											extend_range+=1
											break
										print('out of tries. Bad peak prominences')
										extend_range = 3
										break
								else:
									extend_prom+=1
									extend_x+=1
						else:
							if start_index > min_index_to_plot:
								self.plot_filtering(good_RR_interval_peaks, [], end_index, min_rr_interval, max_rr_interval, [min_y], min_prom, plot_verbosity)
							if (extend_x+extend_y+extend_prom) >= tries-2 or ultra_extend == True:
								ultra_extend = True
								extend_y = 0
								extend_prom = 0
								extend_x+=4
								if extend_x > 60:
									if end_index == len(self.ecg_data)-1:
										extend_range+=1
										break
									print('out of tries. Bad peak heights')
									extend_range = 3
									break
							else:
								extend_prom+=1
								extend_y+=1
								extend_x+=1
					else:
						if start_index > min_index_to_plot:
							self.plot_filtering(good_RR_interval_peaks, [], end_index, min_rr_interval, max_rr_interval, [min_y], min_prom, plot_verbosity)
						if (extend_x+extend_y+extend_prom) >= tries-2 or ultra_extend == True:
							ultra_extend = True
							extend_y = 0
							extend_prom = 0
							extend_x+=4
							if extend_x > 60:
								if end_index == len(self.ecg_data)-1:
									extend_range+=1
									break
								print('out of tries. Bad peak rr-intervals')
								extend_range = 3
								break
						else:
							extend_x+=1
							extend_y+=1

				end_index = self.total_peak_indices[-1]

		#plot before entering calibration mode
		if input_verbosity > 0:
			plt.plot(self.ecg_data[: self.total_peak_indices[-1]+2*self.sampling_rate])
			plt.plot(self.total_peak_indices, self.ecg_data[self.total_peak_indices], 'o')
			plt.legend([self.driver_file_name+' Traversal Complete'])
			plt.show(block = False)

		if extend_range < 3:
			print('traversal complete.')
			if input_verbosity > 0:
				x = ecg_input(message = 'press ENTER to close the window.', end = '')
				if x == '':
					plt.close('all')
					return self.total_peak_indices




		else:
			print('Caliberate again')
			self.calibrate(start_index = end_index, input_verbosity = input_verbosity)
			return self.find_peaks(how_many_previous_peaks_to_use = how_many_previous_peaks_to_use, input_verbosity = input_verbosity, min_index_to_plot = min_index_to_plot, print_verbosity = print_verbosity, plot_verbosity = plot_verbosity)








