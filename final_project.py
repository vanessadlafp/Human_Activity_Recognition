import os
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats
import math
from siml.sk_utils import *
from siml.signal_analysis_utils import *
import sensormotion as sm
from scipy.fftpack import fft
from scipy.signal import welch

from IPython.display import display
import matplotlib.pyplot as plt
%matplotlib inline
import pywt


import time
import datetime as dt
from collections import defaultdict, Counter

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split

import warnings
warnings.filterwarnings("ignore")

# Load and View Input Data
train_ts= pd.read_csv("train_time_series.csv")
train_labels= pd.read_csv("train_labels.csv")
print("Shape of time series signals data:", train_ts.shape)
print("Shape of labels data:", train_labels.shape)

##We create a function that calculates our vectors' magnitude

def magnitude(activity):
    '''
    Calculates the magnitude of a 3D vector
    '''
    x_2= activity['x']*activity['x']
    y_2= activity['y']*activity['y']
    z_2= activity['z']*activity['z']
    mag_2= x_2+y_2+z_2
    mag= mag_2.apply( lambda x: math.sqrt(x))
    
    return mag

train_ts['m']= magnitude(train_ts)


#Time-series Data Visualization

##we put all signals in the same plot
plt.figure()
plt.plot(train_ts['timestamp'], train_ts['x'], linewidth=0.5, color='r', label='x-component')
plt.plot(train_ts['timestamp'], train_ts['y'], linewidth=0.5, color='b', label='y-component')
plt.plot(train_ts['timestamp'], train_ts['z'], linewidth=0.5, color='g', label='z-component')
plt.xlabel('timestamp')
plt.ylabel('acceleration')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
plt.savefig('Time-series Data Visualization.pdf')

##we make now compare them side by side 
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('X-axis: Side to side motion')
ax[0].plot(train_ts['timestamp'], train_ts['x'], linewidth=0.5, color='r')

ax[1].set_title('Y-axis: Up down motion')
ax[1].plot(train_ts['timestamp'], train_ts['y'], linewidth=0.5, color='b')

ax[2].set_title('Z-axis: Forward backward backward')
ax[2].plot(train_ts['timestamp'], train_ts['z'], linewidth=0.5, color='g')

ax[3].set_title('Magnitude, m: Combined X-Y-Z')
ax[3].plot(train_ts['timestamp'], train_ts['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

####Distribution of activities:
activity= {1:'standing', 2: 'walking', 3: 'stairs down', 4: 'stairs up'}
train_labels["activiy"] = train_labels["label"].map(activity)

plt.figure()
plt.title('No of Datapoints per Activity', fontsize=15)
sns.countplot(train_labels["activiy"])
plt.savefig('No of Datapoints per Activity.pdf')


##Stationary and Moving activities are completely different
##merge maggnitutes of acceleration to plot activity's behaviours
act_behaviours= pd.merge(train_labels,train_ts, on='timestamp')

sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(act_behaviours, hue='activiy', size=6,aspect=2)
facetgrid.map(sns.distplot,'m', hist=False)\
    .add_legend()

# for plotting purposes taking datapoints of each activity to a different dataframe
df1 = act_behaviours[act_behaviours['label']==1]
df2 = act_behaviours[act_behaviours['label']==2]
df3 = act_behaviours[act_behaviours['label']==3]
df4 = act_behaviours[act_behaviours['label']==4]

plt.figure(figsize=(14,7))
plt.subplot(2,2,1)
plt.title('Stationary Activities(Zoomed in)')
sns.distplot(df1 ['m'],color = 'r',hist = False, label = 'Standing')
plt.legend(loc='center')

plt.subplot(2,2,2)
plt.title('Moving Activities')
sns.distplot(df2['m'],color = 'red',hist = False, label = 'Walking')
sns.distplot(df3['m'],color = 'blue',hist = False,label = 'Walking Up')
sns.distplot(df4['m'],color = 'green',hist = False, label = 'Walking down')
plt.legend(loc='center right')


plt.tight_layout()
plt.show()

##Magnitude of an acceleration can saperate it well
plt.figure(figsize=(7,7))
sns.boxplot(x= 'label', y='m',data=act_behaviours, showfliers=False, saturation=1)
plt.ylabel('Acceleration Magnitude mean')
plt.show()

##############################################################################
#####Sampling Time and Frequency

##We first convert values in columns UTC time from type str to type datetime
##in pandas to be able to work with their values

train_ts['UTC time']= pd.to_datetime( pd.to_datetime(train_ts['UTC time'],\
        format='%Y-%m-%dT%H:%M:%S.%f'))

train_labels['UTC time']= pd.to_datetime( pd.to_datetime(train_labels['UTC time'],\
        format='%Y-%m-%dT%H:%M:%S.%f'))

# Sampling time and frequency for time series signals 
T_train_ts= np.mean([train_ts['UTC time'].iloc[i+1]-train_ts['UTC time'].iloc[i] for i in range(len(train_ts['UTC time'])-1)]).total_seconds()
f_train_ts= 1/T_train_ts
##T_train_ts= 0.10 s and f_train_ts= 9.98 Hz

# Sampling time and frequency for labels
T_train_label= np.mean([train_labels['UTC time'].iloc[i+1]-train_labels['UTC time'].iloc[i] for i in range(len(train_labels['UTC time'])-1)]).total_seconds()
f_train_label= 1/T_train_label
##T_train_ts= 1.00 s and f_train_ts= 1.00 Hz

################################################################################
##CREATING VECTORS OF 375 DATA POINTS WITH THEIR 10 RESPECTIVE OBSERVATIONS EACH
##TO FORM A  MATRIX OF 375 ROWS, 4 COLUMNS (X,Y,Z,M)
train_x_list= [train_ts.x.iloc[i:i+10] for i in range(len(train_labels))]
train_y_list= [train_ts.y.iloc[i:i+10] for i in range(len(train_labels))]
train_z_list= [train_ts.z.iloc[i:i+10] for i in range(len(train_labels))]
train_m_list= [train_ts.m.iloc[i:i+10] for i in range(len(train_labels))]

train_signals = np.array([train_x_list, train_y_list, train_z_list, train_m_list])

## I get a matrix of shape((4, 375, 10)) order (0,1,2) so I'll now indicate 
## the desired order (1,2,0) -> (375,10,4)

train_signals= np.transpose(train_signals, (1,2,0))
train_labels= np.array(train_labels['label'].astype(int))

##WE NOW SHUFFLED OUR DATA
def randomize(dataset, labels):
   permutation = np.random.permutation(labels.shape[0])
   shuffled_dataset = dataset[permutation, :]
   shuffled_labels = labels[permutation]
   return shuffled_dataset, shuffled_labels

train_signals, train_labels = randomize(train_signals, np.array(train_labels))

###############################################################################
## Frequency Transformation Functions

def get_values(y_values, T, N, f_s):
    y_values = y_values
    x_values = [(1/f_s) * kk for kk in range(0,len(y_values))]
    return x_values, y_values

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values
    
def get_first_n_peaks(x,y,no_peaks=5):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks
    
def get_features_ft(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y
 
def extract_features_labels(dataset, labels, T, N, f_s, denominator):
    percentile = 5
    list_of_features = []
    list_of_labels = []
    for signal_no in range(0, len(dataset)):
        features = []
        list_of_labels.append(labels[signal_no])
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            
            signal_min = np.nanpercentile(signal, percentile)
            signal_max = np.nanpercentile(signal, 100-percentile)
            #ijk = (100 - 2*percentile)/10
            mph = signal_min + (signal_max - signal_min)/denominator
            
            features += get_features_ft(*get_psd_values(signal, T, N, f_s), mph)
            features += get_features_ft(*get_fft_values(signal, T, N, f_s), mph)
            features += get_features_ft(*get_autocorr_values(signal, T, N, f_s), mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)

#####APPLYING AND VISUALIZING FREQUENCY TRANSFORMS 
activities_description = {
    1: 'Standing',
    2: 'Walking',
    3: 'Stairs down',
    4: 'Stairs up'
}

N = 10 ##samples per label
f_s = 1 #1 Hz for train_labels #10 Hz for train_ts 
t_n = 1 #1 sec for train_labels #0.1 sec for train_ts
T = t_n / N #
sample_rate = 1 / f_s
denominator = 10

labels = ['x-component', 'y-component', 'z-component']
colors = ['r', 'g', 'b']
suptitle = "Different signals for the activity: {}"
 
xlabels = ['Time [sec]', 'Freq [Hz]', 'Freq [Hz]', 'Time lag [s]']
ylabel = 'Amplitude'
axtitles = [['Standing: Acc', 'Walking: Acc', 'Stairs dn: Acc', 'Stairs up: Acc'],
            ['Standing: FFT Acc', 'Walking: FFT Acc', 'Stairs dn: FFT Acc', 'Stairs up: FFT Acc'],
            ['Standing: PSD Acc', 'Walking: PSD Acc', 'Stairs dn: PSD Acc', 'Stairs up: PSD Acc'],
            ['Standing: Autocorr Acc', 'Walking: Autocorr Acc', 'Stairs dn: Autocorr Acc', 'Stairs up: Autocorr Acc']
           ]

list_functions = [get_values, get_fft_values, get_psd_values, get_autocorr_values]
signal_no_list = [5, 20, 160, 120]
activity_name = list(activities_description.values())

f, axarr = plt.subplots(nrows=4, ncols=4, figsize=(12,8))
f.suptitle(suptitle.format(activity_name), fontsize=10)
 
for row_no in range(0,4):
    for col_no in range(0,4):
        for comp_no in range(0,3):
            color = colors[comp_no % 3]
            label = labels[comp_no % 3]

            axtitle  = axtitles[row_no][col_no]
            xlabel = xlabels[row_no]
            value_retriever = list_functions[row_no]

            ax = axarr[row_no][col_no]
            ax.set_title(axtitle, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=10)
            
            if col_no == 0:
                ax.set_ylabel(ylabel, fontsize=10)

            signal_no = signal_no_list[col_no]
            signals = train_signals[signal_no, :, :]
            signal_component = signals[:, comp_no]
            x_values, y_values = value_retriever(signal_component, T, N, f_s)
            ax.plot(x_values, y_values, linestyle='-', color=color, label=label)
            
            if row_no > 0:
                max_peak_height = 0.1 * np.nanmax(y_values)
                indices_peaks = detect_peaks(y_values, mph=max_peak_height)
                ax.scatter(x_values[indices_peaks], y_values[indices_peaks], c=color, marker='*', s=60)
            if col_no == 3:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.6)
plt.show()

###############################################################################
#Extract Frequency Transformed Features


X_train_ft, Y_train_ft = extract_features_labels(train_signals, train_labels, T, N, f_s, denominator)
print(X_train_ft.shape)
print(Y_train_ft.shape)

#############################################################################
#Train Classifiers

X_train, X_val, Y_train, Y_val = train_test_split(X_train_ft, Y_train_ft, train_size=0.8, random_state=1)
models = batch_classify(X_train, Y_train, X_val, Y_val)
display_dict_models(models)

##Resampling for Class Imbalance

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_ros, Y_ros = ros.fit_sample(X_train, Y_train)
models = batch_classify(X_ros, Y_ros, X_val, Y_val)
display_dict_models(models)