1. Running % python3 peak2rr.py -outDir rr_data -file driver_data_final/peak_indices/drive*.csv
will convert the peak files into csv files with timestamps & rr intervals — results are in rr_data directory.

2. Running % python3 extract_rr_features.py --out driver_hrv.csv -file rr_data/drive*.csv
will extract the rr_features and produces driver_hrv.csv.

3. Running % python3 driver_gsr.py driver_hrv.csv driver_data_final/drive*.csv
will add the Phasic and Tonic features to the driver_hrv.csv file, and produces driver_hrv_gsr.csv.

4. Running % python3 combine_features_with_raw_data.py
will add the raw data columns (columns in driver_data_final) to each corresponding driver in driver_hrv_gsr.csv and will produce final_file.csv, which we used in training our model with tpot.






