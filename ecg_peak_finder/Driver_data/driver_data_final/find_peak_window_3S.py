import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from  scipy.signal import find_peaks


def find_period_fft(M, min_v = 0, max_v = 400, sampling_rate = 250):
	#L = np.round(L, 1)
	# Remove DC component, as proposed by Nils Werner
	L = M - np.mean(M)
	# Window signal
	#L *= scipy.signal.windows.hann(len(L))


	fft = np.fft.rfft(L, norm="ortho")

	x = np.arange(0, len(fft))


	def abs2(x):
		return x.real**2 + x.imag**2

	selfconvol=np.fft.irfft(abs2(fft), norm="ortho")
	selfconvol=selfconvol/selfconvol[0]


	x = np.arange(0, len(selfconvol))
	# plt.plot(x, selfconvol)
	# plt.title('selfconvol')
	# plt.show()

	# let's get a max, assuming a least 4 periods...
	multipleofperiod=np.argmax(selfconvol[int(sampling_rate/5):len(L)//2]) # TODO fix this
	Ltrunk=L[0:(len(L)//multipleofperiod)*multipleofperiod]

	fft = np.fft.rfft(Ltrunk, norm="ortho")
	selfconvol=np.fft.irfft(abs2(fft), norm="ortho")
	selfconvol=selfconvol/selfconvol[0]
	selfconvol = selfconvol[:max_v]
	selfconvol = np.where(selfconvol<0, 0, selfconvol)
	idx = 1
	while selfconvol[idx-1] > selfconvol[idx] or idx < min_v:  # get rid of starting peak
		selfconvol[idx-1] = 0
		idx += 1
	max_idx = np.argmax(selfconvol)

	return max_idx, selfconvol

def find_period_autocorrelation(norm_sig, fs=250, max_v=400):
	acf = sm.tsa.acf(norm_sig, nlags=len(norm_sig),fft=True)
	lag = np.arange(len(norm_sig)) / fs
	acf = np.where(acf<0, 0, acf) # get rid of negative valuse
	idx = 1
	while acf[idx-1] > acf[idx] or idx < 75:  # get rid of starting peak
		acf[idx-1] = 0
		idx += 1

	# TODO get rid of second peaks (peak that is exactly 2x earlier peak)
	max_idx = np.argmax(acf[:max_v])  

	return max_idx, acf[:max_v]


# qt_dir = "qt-database-1.0.0"
# if not os.path.isdir(qt_dir):
# 	print("Can't find directory:",qt_dir)
# 	sys.exit()

# num_samples = 3750
# fft_diff = 0
# ac_diff = 0
# percent_error = 0.00

# for file in os.listdir(qt_dir):
# 	if not file.endswith("sele0107.atr"): continue
# 	print(file)
# 	path = os.path.join(qt_dir,file)
# 	annotation = wfdb.rdann(path.replace('.atr',''),'atr')
	
# 	all_peaks = []
# 	for a in annotation.__dict__['sample']:
# 		if a < num_samples:
# 			all_peaks.append(a)
# 		else: break

# 	path = os.path.join(qt_dir,file.replace(".atr",""))
# 	record = wfdb.rdrecord(path)

# 	for ecg_idx in range(2):
# 		sig = record.__dict__['p_signal'][:num_samples,ecg_idx]
# 		sig = sig - sig.mean()
# 		sig = -1*sig
# 		fs=record.__dict__['fs']

# 		fft_period, fft_y = find_period_fft(sig)

# 		ac_period, ac_y = find_period_autocorrelation(sig, fs=fs)

# 		real_period = int(np.mean(np.diff(all_peaks)))

# 		fft_diff += abs(real_period - fft_period)
# 		ac_diff += abs(real_period - ac_period)
# 		print("Error FFT",fft_diff,"AC",ac_diff)

# 		if abs(real_period - ac_period) > real_period*percent_error or abs(real_period - fft_period) > real_period*percent_error:
# 			peaks, _ = find_peaks(sig, prominence=(0.5,None), height=0.1, distance=fft_period*.7)
# 			ac_percent = np.round((abs(real_period - ac_period) / real_period) * 100,1)
# 			fft_percent = np.round((abs(real_period - fft_period) / real_period) * 100,1)
# 			print("Real:{:}  AC:{:} {:}  FFT {:} {:}%".format(real_period, ac_period, ac_percent, fft_period, fft_percent))

# 			fig,axes = plt.subplots(nrows=3, ncols=1, figsize=(6,9))
# 			fig.suptitle(file.replace('.atr','') + " : " + str(ecg_idx) )
# 			text = "Norm {:} samples ".format(real_period)
# 			x = np.arange(0, len(sig))
# 			axes[0].plot(x, sig)
# 			axes[0].set_title(text)
# 			axes[0].plot(x[all_peaks], sig[all_peaks], 'bo')
# 			axes[0].plot(peaks, sig[peaks], "rx")

# 			text = "Autocorrelation {:} samples".format(ac_period)
# 			x = np.arange(0, len(ac_y))
# 			axes[1].plot(x, ac_y)
# 			axes[1].set_title(text)
# 			axes[1].plot(x[ac_period], ac_y[ac_period], 'bo')

# 			text = "FFT {:} samples".format(fft_period)
# 			x = np.arange(0, len(fft_y))
# 			axes[2].plot(x, fft_y)
# 			axes[2].set_title(text)
# 			axes[2].plot(x[fft_period], fft_y[fft_period], 'ro')
# 			plt.show()



