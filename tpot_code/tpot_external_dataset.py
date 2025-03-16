import caffeine
import sys
import statistics
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from scipy import signal
from sklearn.model_selection import train_test_split


### Ottesen's code with moinor adjustments. Adjusted lines are commented in each line.
dataframe_hrv = pd.read_csv('dataframe_hrv.csv')

dataframe_hrv = dataframe_hrv.reset_index(drop=True)


def fix_stress_labels(df='',label_column='stress'):
	df['stress'] = np.where(df['stress']>=0.5, 1, 0)
	return df
dataframe_hrv = fix_stress_labels(df=dataframe_hrv)


def missing_values(df):
    df = df.reset_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df[~np.isfinite(df)] = np.nan
    df.plot( y=["HR"])
    df['HR'].fillna((df['HR'].mean()), inplace=True)
    df['HR'] = signal.medfilt(df['HR'],13) 
    df.plot( y=["HR"])

    df=df.fillna(df.mean()) # removed inplace = True, which resulted in return value of None (bug fix)
    return df


dataframe_hrv = missing_values(dataframe_hrv)





selected_x_columns = ['HR','interval in seconds','AVNN', 'RMSSD', 'pNN50', 'TP', 'ULF', 'VLF', 'LF', 'HF','LF_HF']

X = dataframe_hrv[selected_x_columns]
y = dataframe_hrv['stress']

def do_tpot(generations=5, population_size=10,X='',y=''):

    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.80,test_size=0.20)
    tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2,cv=3, random_state=19) #added random_state = 19
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_pipeline_Ottesen.py')
    return tpot

tpot_classifer = do_tpot(generations=100, population_size=20,X=X,y=y)




###






### my (Ala Kaymaram's) code


# determine stress the way Ottesen did above:
# "The median of the galvanic skin response value is taken as the cut-off point to determine the stressed state,
# any value above the median value is labelled as stress, and any value below the median value is labelled as not stressed"

def Ottesen_polarize(label_data = None):
	med = statistics.median(label_data)
	polarized_data = [1 if x >= med else 0 for x in label_data]
	return polarized_data


def Our_method_polarize(label_data = None, threshhold = 3):
    return [1 if float(x) > threshhold else 0 for x in label_data] # 1 stressed, 0 not stressed



input_csv = pd.read_csv('final_file.csv')

selected_features = ['ECG', 'HR', 'RESP', 'mean_nni', 'sdnn', 'sdsd', 'nni_50', 'pnni_50','nni_20','pnni_20','rmssd','median_nni']
selected_features.extend(['range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'std_hr', 'lf','hf', 'lf_hf_ratio','lfnu','hfnu','total_power', 'vlf'])

selected_features_and_values = input_csv[selected_features]
selected_label = 'phasic_peaks_over_1'

selected_label_and_values = (input_csv[selected_label]).to_numpy()




# if using ottesen's
#selected_label_and_values_polarized = Ottesen_polarize(selected_label_and_values)

# if using our method
selected_label_and_values_polarized = Our_method_polarize(selected_label_and_values, threshhold = 1)




features = selected_features_and_values
label = selected_label_and_values_polarized
X_train, X_test, y_train, y_test = train_test_split(features, label, train_size=0.80, test_size=0.20)


tpot = TPOTClassifier(generations=10, population_size=100,
offspring_size=None, mutation_rate=0.9,
crossover_rate=0.1,
scoring='accuracy', cv=5,
subsample=1.0, n_jobs=-1,
max_time_mins=None, max_eval_time_mins=5,
random_state=19, config_dict='TPOT light',
template=None,
warm_start=True,
memory='auto',
use_dask=False,
periodic_checkpoint_folder=None,
early_stop=None,
verbosity=2,
disable_update_check=False,
log_file=None
)


tpot.fit(X_train, y_train)


tpot.export('tpot_pipeline.py')
score1 = tpot.score(X_test, y_test)
score2 = tpot.score(X_train, y_train)
print("Model's score on the test set:",score1)
print("Model's score on the training set:",score2)