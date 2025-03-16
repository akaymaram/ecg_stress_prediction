import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import savgol_filter, find_peaks, peak_prominences
import find_peak_window_3S
import statistics
import math



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

def plot_horizontal_line(list_of_y_values = [], color = 'r'):
	for x in list_of_y_values: plt.axhline(x, c = color)

def plot_vertical_line(list_of_x_values = [], color = 'r'):
	for x in list_of_x_values: plt.axvline(x, c = color)

def plot_dots(x_values = [], y_values = [], color = 'tab:orange'):
	if len(x_values) != len(y_values):
		print('bad input lists. list lengths do not match.')
		return None
	index = 0
	while index < len(x_values):
		plt.plot(x_values[index], y_values[index], '.', c = color)
		index+=1

def plot_points(x_values = [], y_values = [], shape = 'x', color = 'tab:orange'): plt.plot(x_values, y_values, shape, c = color)


def plot_points_connected(x_values = [], y_values = [], color = '#1f77b4'):
	plt.plot(x_values, y_values, c = color)


class Ecg:
	def __init__(self, file_name, sampling_rate = 250, timeCol = 1, ecgCol = 2, has_header = True):
		self.sampling_rate = int(sampling_rate)
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
		self.raw_ecg_data = (dataframe.iloc[: ,ecgCol]).to_numpy()
		self.ecg_data = self.raw_ecg_data
		self.ecg_data_min = self.ecg_data.min()
		self.ecg_data_mean = self.ecg_data.mean()
		self.ecg_data_max = self.ecg_data.max()
		self.size = len(self.ecg_data)
		self.time_data = np.array(dataframe.iloc[: ,timeCol])



	def update_attributes(self):
		self.ecg_data_min = self.ecg_data.min()
		self.ecg_data_mean = self.ecg_data.mean()
		self.ecg_data_max = self.ecg_data.max()



	def normalize(self):
		count = 0
		temp_list = []
		max_min = self.ecg_data_max - self.ecg_data_min
		while count < len(self.ecg_data):
			temp_list.append(((self.ecg_data[count] - self.ecg_data_min)/max_min))
			count+=1
		self.ecg_data = np.array(temp_list)


		#self.ecg_data = ((self.ecg_data - self.ecg_data_min)/(self.ecg_data_max - self.ecg_data_min))
		self.update_attributes()

	def do_smoothing(self):
		window = self.sampling_rate
		if self.sampling_rate % 2 == 0:
			window += 1
		self.ecg_data = self.ecg_data - savgol_filter(self.ecg_data, window, 3)
		self.update_attributes()

	def invert(self):
		self.ecg_data = (self.ecg_data * -1)+1
		self.update_attributes()
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
		self.update_attributes()

		# another idea for auto invertion
		# segment = self.ecg_data[: 6*self.sampling_rate]
		# if  max(segment) - statistics.median(segment) < statistics.median(segment) - min(segment):
		# 	print('top dis:', max(segment) - statistics.median(segment))
		# 	print('bottom dis:', statistics.median(segment) - min(segment))
		# 	self.invert()




	def scale_outliers(self):
		mean = statistics.mean(self.ecg_data)
		std = statistics.stdev(self.ecg_data)
		max_end = mean+3*std
		min_end = mean-2*std
		x = 0
		while x < self.size:
			value = self.ecg_data[x]
			if value > max_end: self.ecg_data[x] = max_end
			elif value < min_end: self.ecg_data[x] = min_end
			x+=1
		self.update_attributes()


	def calibrate(self, start_index = 0, input_verbosity = 1, plot_list_of_peaks_one_by_one = False, num_secconds_in_calibration = 1):

		if start_index == 0:
			start_index = math.ceil(self.sampling_rate/5)
		num_secconds_in_calibration = num_secconds_in_calibration
		end_index = start_index + self.sampling_rate*num_secconds_in_calibration
		if end_index+self.sampling_rate >= self.size: return 0
		max_heart_rate_bpm = 180
		min_heart_rate_bpm = 30
		max_number_of_peaks = int((max_heart_rate_bpm/60)*num_secconds_in_calibration)
		min_distance = self.sampling_rate*60/min_heart_rate_bpm
		min_height = self.ecg_data[start_index:end_index].max()-.1
		min_prominence = .5
		count = 0
		segment_peaks = None
		while True:
			segment_peaks, _ = find_peaks(self.ecg_data[start_index: end_index], distance = min_distance, height = min_height, prominence = min_prominence)
			prom,_,_ = peak_prominences(self.ecg_data[start_index:end_index], segment_peaks)
			segment_peaks += start_index
			actual_prominence = self.ecg_data[segment_peaks].tolist() - prom
			if plot_list_of_peaks_one_by_one:
				if len(segment_peaks) > 0: plot_dots(segment_peaks, [y-min_prominence for y in self.ecg_data[segment_peaks]], color = 'r')
				vertical_line =[]
				color = ['r', 'y', 'b', 'k']
				counter = -1
				if len(segment_peaks) > 0:
					for x in segment_peaks:
						counter+=1
						if counter == 5: counter = 0
						vertical_line = [x-min_distance, x+min_distance]
						plot_vertical_line(vertical_line, color[counter])
					
				plot_horizontal_line([min_height])
				end_index_plot = end_index
				if len(segment_peaks) > 0:
					plot_points(segment_peaks, self.ecg_data[segment_peaks])
					end_index_plot = int(max(end_index_plot, max(segment_peaks),max(vertical_line)))
				if end_index_plot >= self.size-1:
					print('traversal complete.')
					if input_verbosity > 0:
						x = ecg_input(message = 'press ENTER to close the window.', end = '')
					if x == '':
						plt.close('all')
						return self.total_peak_indices



				x_vals = [x for x in range(start_index,end_index_plot)]
				y_vals = self.ecg_data[start_index:end_index_plot]
				plot_points_connected(x_vals,y_vals, color = 'g')
				if len(segment_peaks) > 0: plot_dots(segment_peaks, actual_prominence, color = 'g')
				plt.show(block=False)
				x = ecg_input(message = 'press ENTER for next.')
				if x == '': plt.close('all')
			if len(segment_peaks) > 2: break


			last_index = 0
			if len(self.total_peak_indices) > 0: last_index = self.total_peak_indices[-1]
			dist = int((start_index - last_index)/self.sampling_rate)
			if dist > 5 and len(segment_peaks) < 1:
				self.calibrate(start_index+math.ceil(dist*self.sampling_rate/5), input_verbosity, plot_list_of_peaks_one_by_one, 1)
				return 0


			if count > 10:
				if num_secconds_in_calibration < 4:
					self.calibrate(start_index, input_verbosity, plot_list_of_peaks_one_by_one, num_secconds_in_calibration+1)
				else:
					self.calibrate(start_index+int(self.sampling_rate/2), input_verbosity, plot_list_of_peaks_one_by_one, 1)
				return 0
			
			if (count % 3) == 2:
				min_distance = max(min_distance-math.ceil(self.sampling_rate/2), 1)
			elif (count % 3) == 1:
				min_prominence = max(min_prominence-.05, 0)
			elif (count % 3) == 0:
				min_height = max(min_height-.05, 0)
			count+=1
		




		no_repeat_list_of_tuple_of_peaks = [segment_peaks]

		good_height_peaks = []
		for tupl in no_repeat_list_of_tuple_of_peaks:
			for peak in tupl:
				bad_height = False
				if self.ecg_data[peak] < .6*max(self.ecg_data[start_index:end_index]):
					bad_height = True
					break
			if bad_height == True: continue
			good_height_peaks.append(tupl)



		
		no_repeat_array_peaks = np.array(good_height_peaks)






		#step-by-step testing
		# if plot_list_of_peaks_one_by_one == True:
		# 	for tup in no_repeat_array_peaks:
		# 		plot_points_connected([x for x in range(start_index,end_index)], self.ecg_data[start_index: end_index])
		# 		plot_points(list(tup),self.ecg_data[list(tup)])
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
					plot_points_connected([x for x in range(start_index,end_index)], self.ecg_data[start_index: end_index])
					the_list = list(no_repeat_array_peaks[x]) #  from tuple to list
					#plt.plot(the_list, self.ecg_data[the_list], 'x')
					plot_points(the_list, self.ecg_data[the_list])
					plt.legend([self.driver_file_name+' good_RR_interval_peaks'])
					plt.show(block = False)
					user_input = ecg_input(message = 'press ENTER for next.')
					if user_input == '':
						plt.close()

			for index in good_RR_interval_peaks:
				prom,_,_ = peak_prominences(self.ecg_data[: end_index+int(self.sampling_rate/2)], no_repeat_array_peaks[index])
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
				plot_points_connected([x for x in range(start_index, end_index)], self.ecg_data[start_index: end_index])
				plot_points(segment_peaks, self.ecg_data[segment_peaks], 'o')
				plt.legend([self.driver_file_name, message])
				plt.show(block = False)
				x = ecg_input(['', ' '], message = 'Press ENTER to skip half a sampling rate or SPACE ENTER to expand the calibration window.')
				if x == '':
					plt.close('all')
					self.calibrate(start_index+int(self.sampling_rate/2), input_verbosity, plot_list_of_peaks_one_by_one, num_secconds_in_calibration)
					return 1
				elif x == ' ':
					plt.close('all')
					self.calibrate(start_index, input_verbosity, plot_list_of_peaks_one_by_one, num_secconds_in_calibration+1)
					return 1
			else:
				plt.close('all')
				self.calibrate(start_index+int(self.sampling_rate/2), input_verbosity, plot_list_of_peaks_one_by_one, num_secconds_in_calibration)
				return 1




		#fft
		
		fft_period, fft_y = find_peak_window_3S.find_period_fft(self.ecg_data[start_index: end_index], math.ceil(self.sampling_rate/3), math.ceil(1.5*self.sampling_rate), self.sampling_rate)
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
			plot_points_connected([x for x in range(start_index, end_index)], self.ecg_data[start_index: end_index])
			plot_points(segment_peaks, self.ecg_data[segment_peaks], 'o')
			plt.legend([str(self.driver_file_name)+' calibration mode'])
			plt.show(block=False)
			x = ecg_input(['', 's', 'x'], message = 'press ENTER for good peaks, S to skip half a sampling rate, and X expand the calibration window.', end = '')

		if x == '':
			prom,L,R = peak_prominences(self.ecg_data[: end_index], segment_peaks)
			self.total_peak_indices.extend(segment_peaks)
			self.total_peak_raw_ecg_values.extend(self.raw_ecg_data[segment_peaks])
			self.total_peak_ecg_values.extend(self.ecg_data[segment_peaks])
			self.total_peak_prominences.extend(prom)
			self.total_peak_timestamps.extend(initial_peak_timestamps)
			self.total_RR_intervals.extend(initial_RR_intervals)
			plt.close('all')
			return 1
		elif x.lower() == 's':
			plt.close('all')
			self.calibrate(start_index = start_index+int(self.sampling_rate/2), input_verbosity = input_verbosity, num_secconds_in_calibration = num_secconds_in_calibration)
			return 1
		elif x.lower() == 'x':
			plt.close('all')
			self.calibrate(start_index, input_verbosity, plot_list_of_peaks_one_by_one, num_secconds_in_calibration+1)
			return 1
		else: print('bad input')


	def calib(self, min_distance = 1, start_index = 0, height = 0, prominence = 0):
		segment_peaks, _ = find_peaks(self.ecg_data[start_index: end_index], distance = min_distance, height = height, prominence = prominence)

		initial_peak_timestamps = np.array(self.time_data[segment_peaks])
		initial_RR_intervals = np.diff(initial_peak_timestamps)

		
		fft_period, fft_y = find_peak_window_3S.find_period_fft(self.ecg_data[0: 1000], math.ceil(self.sampling_rate), math.ceil(1.5*self.sampling_rate), self.sampling_rate)
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
		plt.plot([x for x in range(start_index,end_index)],self.ecg_data[start_index: end_index])
		x = 0
		if len(list_of_labels) > 0:
			if len(list_of_labels) == len(list_of_points):
				while x < len(list_of_points):
					point = list_of_points[x]
					plt.plot(point, self.ecg_data[point], 'o', label = list_of_labels[x], color = 'g')
					x+=1
					plt.legend()
			else:
				while x < len(list_of_points):
					point = list_of_points[x]
					plt.plot(point, self.ecg_data[point], 'o', label = list_of_labels[0], color = 'g')
					x+=1
					plt.legend()

		else:
			while x < len(list_of_points):
				point = list_of_points[x]
				plt.plot(point, self.ecg_data[point], 'x', color = 'r')
				x+=1
		plt.show(block = False)


	def plot_filtering(self, list_of_points = [], list_of_vertical_lines = [], list_of_horizontal_lines= [], list_of_dots = [[],[]], plot_verbosity = 1):
		end_index = 0
		if len(list_of_points) != 0: end_index = max(list_of_points)
		if len(list_of_vertical_lines) != 0: end_index = int(max(max(list_of_vertical_lines), end_index))

		if plot_verbosity < 1:
			if len(list_of_points) == 1: return 0 #don't plot
		start = -9
		if len(self.total_peak_indices) < 10: start = 0
		if len(self.total_peak_indices) != 0:
			last_peak_timestamp = self.total_peak_timestamps[-1]
			last_peak_index = self.total_peak_indices[-1]
			print('last_peak_timestamp:', last_peak_timestamp)
		

		#ecg, previous peaks (o), possible peaks (x)
		start_index = 0
		if len(self.total_peak_indices) > 0:
			start_index = max(self.total_peak_indices[start] - self.sampling_rate, 0)
			plot_points(self.total_peak_indices[start:], self.ecg_data[self.total_peak_indices[start:]], 'o')
		plot_points_connected([x for x in range(start_index,end_index)], self.ecg_data[start_index: end_index], color = 'g')
		plot_points(list_of_points, self.ecg_data[list_of_points], 'x')


		plot_vertical_line(list_of_vertical_lines)
		plot_horizontal_line(list_of_horizontal_lines)
		plot_dots(list_of_dots[0],list_of_dots[1])

		plt.show(block=False)

		x = ecg_input(message = 'press ENTER for next.')
		if x == '':
			plt.close('all')





	def find_all_peaks(self,  how_many_previous_peaks_to_use = 1, input_verbosity = 1, min_index_to_plot = -1, print_verbosity = 0, plot_verbosity = 1):

		if len(self.total_peak_indices) == 0:
			self.calibrate(input_verbosity = input_verbosity)
		print('working on it...')

		num_peak_to_use = min(max(0, len(self.total_peak_indices)), how_many_previous_peaks_to_use)*(-1)


		if min_index_to_plot < 0: min_index_to_plot = self.size-1

		extend_range = 1
		end_index = self.total_peak_indices[-1]
		while (end_index < self.size) and extend_range == 1:
			end_index = self.total_peak_indices[-1]
			start_index = end_index
			while end_index < start_index+self.sampling_rate*6:
				end_index+=1
				if end_index == self.size-1:
					break	
			if print_verbosity > 0 or start_index > min_index_to_plot:
				print('last peak index:', start_index)

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

			segment_peaks, _ = find_peaks(self.ecg_data[start_index: end_index], distance = int(self.sampling_rate/4), prominence = p_median/4)
			segment_peaks += start_index

			# plot all raw caught peaks (before filtering)
			#if start_index > min_index_to_plot: self.plot_index(segment_peaks)

			the_peak = None
			found_peak = False
			ultra_extend = False
			extend_x = 0
			extend_y = 0
			extend_prom =0
			tries = 29

			if(len(segment_peaks) == 0):
				if end_index != self.size-1:
					print('no peaks found at all')
					extend_range = 3
				else:
					break
			if True:
				start = -(how_many_previous_peaks_to_use)
				if len(self.total_peak_indices) < how_many_previous_peaks_to_use: start = 0
				while ((extend_x+extend_y+extend_prom) < tries or extend_x < 60) and found_peak == False:
					if print_verbosity > 0 or start_index > min_index_to_plot:
						print('tries:' ,extend_x+extend_y+extend_prom)
						print('extend_x:', extend_x)
					min_rr_interval = max(min(x_median-(2*x_std*(extend_x+1)), x_median*(0.65-.03*extend_x)), x_median*0.5)
					max_rr_interval = max(x_median+(3.5*x_std*(extend_x+1)), x_median*(1.2+.3*extend_x))
					min_RR_index = int(self.total_peak_indices[-1] + min_rr_interval*self.sampling_rate)
					max_RR_index = int(self.total_peak_indices[-1] + max_rr_interval*self.sampling_rate)
					end_index = max(max_RR_index, end_index)
					min_y = max(min(h_median-(1*h_std+1.5*h_std*(extend_y)), h_median*(.7-.2*extend_y)), h_median*.3)
					min_prom = max(min(p_median-(1*p_std+p_std*(extend_prom)), p_median*(.6-.1*extend_prom)), p_median*.25)

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
							prom,L,R = peak_prominences(self.ecg_data[: end_index], good_RR_interval_AND_height_peaks)
							
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
								prom,L,R = peak_prominences(self.ecg_data[: end_index], good_RR_interval_AND_height_AND_prominence_peaks)


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
								# 	return self.find_all_peaks(how_many_previous_peaks_to_use = how_many_previous_peaks_to_use, input_verbosity = input_verbosity, min_index_to_plot = min_index_to_plot, print_verbosity = print_verbosity, plot_verbosity = plot_verbosity)


								#pick first peak
								max_prom = prom[0]
								the_peak_index = min(good_RR_interval_AND_height_AND_prominence_peaks)



								if start_index > min_index_to_plot:
									actual_prominence = self.ecg_data[good_RR_interval_AND_height_AND_prominence_peaks].tolist() - prom
									plot_dots(good_RR_interval_AND_height_AND_prominence_peaks, [y-min_prom for y in self.ecg_data[good_RR_interval_AND_height_AND_prominence_peaks]], color = 'r')
									plot_points(good_RR_interval_AND_height_AND_prominence_peaks, self.ecg_data[good_RR_interval_AND_height_AND_prominence_peaks], 'x')
									plot_points(self.total_peak_indices[start:],self.ecg_data[self.total_peak_indices[start:]], 'o')
									start_range = self.total_peak_indices[start]
									end_range = good_RR_interval_AND_height_AND_prominence_peaks[-1]
									x_list = [x for x in range(start_range, end_range+self.sampling_rate)]
									y_list = self.ecg_data[start_range: end_range+self.sampling_rate]
									plot_points_connected(x_list, y_list)
									plot_vertical_line([min_RR_index, max_RR_index])
									plot_horizontal_line([min_y])
									plot_dots(good_RR_interval_AND_height_AND_prominence_peaks,actual_prominence)
									plt.legend([str(self.driver_file_name)])
									plt.show(block=False)
									x = ecg_input(message = 'press ENTER for next.')
									if x == '': plt.close('all')





								self.total_peak_prominences.append(max_prom)
								self.total_peak_indices.append(the_peak_index)
								self.total_peak_raw_ecg_values.append(self.raw_ecg_data[the_peak_index])
								self.total_peak_ecg_values.append(self.ecg_data[the_peak_index])
								self.total_peak_timestamps.append(self.time_data[the_peak_index])
								self.total_RR_intervals.append(self.total_peak_timestamps[-1]-self.total_peak_timestamps[-2])
								found_peak = True
							else:
								if start_index > min_index_to_plot:
									start_index = math.floor(self.total_peak_indices[-1] + min_rr_interval*self.sampling_rate)
									actual_prominence = self.ecg_data[good_RR_interval_AND_height_peaks].tolist() - prom
									plot_dots(good_RR_interval_AND_height_peaks, [y-min_prom for y in self.ecg_data[good_RR_interval_AND_height_peaks]], color = 'r')
									plot_points(good_RR_interval_AND_height_peaks, self.ecg_data[good_RR_interval_AND_height_peaks], 'x')
									plot_points(self.total_peak_indices[start:],self.ecg_data[self.total_peak_indices[start:]], 'o')
									x_list = [x for x in range(self.total_peak_indices[start], good_RR_interval_AND_height_peaks[-1]+self.sampling_rate)]
									y_list = self.ecg_data[self.total_peak_indices[start]: good_RR_interval_AND_height_peaks[-1]+self.sampling_rate]
									plot_points_connected(x_list, y_list)
									plot_vertical_line([min_RR_index, max_RR_index])
									plot_horizontal_line([min_y])
									plot_dots(good_RR_interval_AND_height_peaks,actual_prominence)
									plt.legend([str(self.driver_file_name)])
									plt.show(block=False)
									x = ecg_input(message = 'press ENTER for next.')
									if x == '': plt.close('all')
								if (extend_x+extend_y+extend_prom) >= tries-2 or ultra_extend == True:
									ultra_extend = True
									extend_y = 0
									extend_prom = 0									
									extend_x+=4
									if extend_x > 60:
										if end_index == self.size-1:
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
									plot_points(good_RR_interval_peaks, self.ecg_data[good_RR_interval_peaks], 'x')
									plot_points(self.total_peak_indices[start:],self.ecg_data[self.total_peak_indices[start:]], 'o')
									x_list = [x for x in range(self.total_peak_indices[start], good_RR_interval_peaks[-1]+self.sampling_rate)]
									y_list = self.ecg_data[self.total_peak_indices[start]: good_RR_interval_peaks[-1]+self.sampling_rate]
									plot_points_connected(x_list, y_list)
									plot_vertical_line([min_RR_index, max_RR_index])
									plot_horizontal_line([min_y])
									plt.legend([str(self.driver_file_name)])
									plt.show(block=False)
									x = ecg_input(message = 'press ENTER for next.')
									if x == '': plt.close('all')
							if (extend_x+extend_y+extend_prom) >= tries-2 or ultra_extend == True:
								ultra_extend = True
								extend_y = 0
								extend_prom = 0
								extend_x+=4
								if extend_x > 60:
									if end_index == self.size-1:
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
							plot_points(segment_peaks, self.ecg_data[segment_peaks], 'x')
							plot_points(self.total_peak_indices[start:],self.ecg_data[self.total_peak_indices[start:]], 'o')
							end_range = max_RR_index
							if len(segment_peaks) > 0: end_range = max(segment_peaks[-1]+self.sampling_rate, end_range)
							x_list = [x for x in range(self.total_peak_indices[start], end_range)]
							y_list = self.ecg_data[self.total_peak_indices[start]: end_range]
							plot_points_connected(x_list, y_list)
							plot_vertical_line([min_RR_index, max_RR_index])
							plot_horizontal_line([min_y])
							plt.legend([str(self.driver_file_name)])
							plt.show(block=False)
							x = ecg_input(message = 'press ENTER for next.')
							if x == '': plt.close('all')							
						if (extend_x+extend_y+extend_prom) >= tries-2 or ultra_extend == True:
							ultra_extend = True
							extend_y = 0
							extend_prom = 0
							extend_x+=4
							if extend_x > 60:
								if end_index == self.size-1:
									extend_range+=1
									break
								print('out of tries. Bad peak rr-intervals')
								extend_range = 3
								break
						else:
							extend_x+=1
							extend_y+=1	


		if extend_range < 3 or start_index + 5*self.sampling_rate >= self.size:
			plt.plot(self.ecg_data[: self.total_peak_indices[-1]+2*self.sampling_rate])
			plt.plot(self.total_peak_indices, self.ecg_data[self.total_peak_indices], 'o')
			plt.legend([self.driver_file_name+' Traversal Complete'])
			plt.show(block = False)
			print('traversal complete.')
			x = ''
			if input_verbosity > 0:
				x = ecg_input(message = 'press ENTER to close the window.', end = '')
			if x == '':
				plt.close('all')
				return self.total_peak_indices

		else:
			print('Caliberate again')
			self.calibrate(start_index = end_index, input_verbosity = input_verbosity)
			return self.find_all_peaks(how_many_previous_peaks_to_use = how_many_previous_peaks_to_use, input_verbosity = input_verbosity, min_index_to_plot = min_index_to_plot, print_verbosity = print_verbosity, plot_verbosity = plot_verbosity)








