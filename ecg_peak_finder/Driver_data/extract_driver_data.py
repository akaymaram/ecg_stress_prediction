import os

for x in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17a','17b']:
	print(x)
	commmand = "rdsamp -r Driver_raw_data/drive" + x + " " + "-p -v -c >drive" + x +".csv"
	os.system(commmand)
