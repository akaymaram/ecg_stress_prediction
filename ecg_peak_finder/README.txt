1. Driver_data

	1. driver_data_final contains:
		the 18 driver csv files,
		Ecg17S.py,
		EcgRunner7S.py,
		find_peak_window_3S.py, and
		a directory called peak_indices, which contains 18 csv files (each having the peak indices and the raw ECG values for the corresponding driver).
	
	2. Running % python3 EcgRunner7S.py d*
	will extract the peaks for each driver file one by one and store them in the directory peak_indices.

	3. Driver_raw_data directory contains the raw sensor data for each driver that can be extracted to csv format using WFDB tools.

	4. extract_driver_data.py uses WFDB commands to extract the driver_raw_data to csv format.


1. QT_data

	1. QT_sample_final contains:
		the 105 samples (timestamps and 2 ECG data columns),
		Ecg14_QT_Optimized.py,
		EcgRunner4_QT_Optimized.py,
		find_peak_window_2.py,
		a directory called peak_indices, which contains 105 csv files (each having the peak indices and the raw egg value for the corresponding driver),and
		a directory called annotation_files, containing the indices and timestamps of marked peaks for all the samples that possess atr annotation (82 total).

	2. Running % python3 EcgRunner4_QT_Optimized.py s*
	will extract the peaks for each sample one by one and store them in the directory peak_indices.

	3. Running % python3 extract_QT_signaldata.py QT_raw_data/*
	will extract the signal driver_raw_data to csv format using WFDB commands.

	4. Running % python3 extract_QT_annotation_data.py QT_raw_data/*
	will extract the atr annotations (of drivers that do have an atr annotation, 82 drivers out of 105) to csv format using WFDB commands in a directory called annotation_files.


