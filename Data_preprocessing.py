import numpy as np
import pandas as pd
import math

from scipy.fftpack import fft
from scipy.signal import welch

# Load and Input Data
train_ts = pd.read_csv("train_time_series.csv")
train_labels = pd.read_csv("train_labels.csv")

test_ts = pd.read_csv("test_time_series.csv")
test_labels = pd.read_csv("test_labels.csv")


def magnitude(activity):
    '''
    Calculates the magnitude of a 3D vector
    '''
    x_2 = activity['x'] * activity['x']
    y_2 = activity['y'] * activity['y']
    z_2 = activity['z'] * activity['z']
    mag_2 = x_2 + y_2 + z_2
    mag = mag_2.apply(lambda x: math.sqrt(x))

    return mag


train_ts['m'] = magnitude(train_ts)

# Parsing of  'UTC Time' column from str to timestamp

train_ts['UTC time'] = train_ts['UTC time'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S.%f'))
train_labels['UTC time'] = train_labels['UTC time'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S.%f'))


# Sampling time and frequency for time series signals
T_train_ts = np.mean([train_ts['UTC time'].iloc[i+1]-train_ts['UTC time'].iloc[i] for i in range(len(train_ts['UTC time'])-1)])
f_train_ts = 1/ T_train_ts.total_seconds()

print("Time-series signals - \nAverage time per sample: {:.2f} seconds\nFrequency per sample: {:.2f} Hz".\
      format(T_train_ts.total_seconds(), f_train_ts))

# Sampling time and frequency for time series labels
T_train_label = np.mean([train_labels['UTC time'].iloc[i+1]-train_labels['UTC time'].iloc[i] for i in range(len(train_labels['UTC time'])-1)])
f_train_label = 1/ T_train_label.total_seconds()

print("\nActivity labels - \nAverage time per sample: {:.2f} seconds\nFrequency per sample: {:.2f} Hz".\
      format(T_train_label.total_seconds(), f_train_label))

# Creating Signals and Labels Arrays for train data

train_x_list = [train_ts.x.iloc[start:start+10] for start in range(len(train_labels))]
train_y_list = [train_ts.y.iloc[start:start+10] for start in range(len(train_labels))]
train_z_list = [train_ts.z.iloc[start:start+10] for start in range(len(train_labels))]
train_m_list = [train_ts.m.iloc[start:start+10] for start in range(len(train_labels))]

train_signals = np.transpose(np.array([train_x_list, train_y_list, train_z_list, train_m_list]), (1, 2, 0))
train_labels = np.array(train_labels['label'].astype(int))

[no_signals_train, no_steps_train, no_components_train] = np.shape(train_signals)


print("The train signals array contains {} signals, each one of length {} and {} components ".format(no_signals_train, no_steps_train, no_components_train))
print("The train labels array contains {} labels".format(np.shape(train_labels)[0]))


# We now randomize data to avoid possible patterns in data to affect our results

def randomize(dataset, labels):
   permutation = np.random.permutation(labels.shape[0])
   shuffled_dataset = dataset[permutation, :]
   shuffled_labels = labels[permutation]
   return shuffled_dataset, shuffled_labels

train_signals, train_labels = randomize(train_signals, np.array(train_labels))

#####################################################################################################################

# The following functions will be applied to the signals to transform them from the time-domain to the frequency-domain
# and give us their frequency spectrum:
# the Fast Fourier Transform (FFT), the Power Spectral Density and the auto-correlation (calculates the
# correlation of a signal with a time-delayed version of itself).

# The Wavelet Transform was not included due to it is better suited for analyzing signals with a dynamic frequency spectrum,
# and that is not the case of our data.


# Fast Fourier Transform (FFT)
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


# Power Spectral Density (PSD)
def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

# Auto-correlation
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]

def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values


t_n = 10
N = 1000
T = t_n / N
f_s = 1/T