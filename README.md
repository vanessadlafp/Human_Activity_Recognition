# Project Overview

This is the capstone project for the HarvardX PH526x- Using Python for Research course.

* Created a tool that predicts the type of physical activity (e.g., walking, climbing stairs) from tri-axial smartphone accelerometer data with a  precision of  93.74%.  (ERROR ~ 0.1618 degrees).
* Engineered features from time series signals on python.
* Optimized (Logistic, Gradient Boosting, Random Forest, Nearest Neighbors and Decision Tree) classifiers using GridsearchCV to reach the best model and train time. 
* Built a client facing API using flask

## Code and Resources Used
Python Version: 3.7
Packages: pandas, numpy, sklearn, matplotlib, seaborn, flask, json, pickle, siml, detecta, imblearn, pprint, collections.

* [Machine Learning with Signal Processing Techniques Github](https://github.com/akhuperkar/HAR-Smartphone-Accelerometer)
* [Machine Learning with Signal Processing Techniques Article](https://ataspinar.com/2018/04/04/machine-learning-with-signal-processing-techniques/)
* [Flask Productionization](https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2) 
* [Flask Tutorial](https://www.youtube.com/watch?v=nUOh_lDMHOU&t=1740s)

## Dataset
The input data used for training in this project consists of two files. The first file, train_time_series.csv, contains the raw accelerometer data, which has been collected using the [Beiwe research platform](https://www.hsph.harvard.edu/onnela-lab/beiwe-research-platform/), and it has the following format: timestamp, UTC time, accuracy, and measurements of linear acceleration along each of the three orthogonal axes x, y, z.

The second file, train_labels.csv, contains the activity labels  numbered with integers:
* 1 = standing
* 2 = walking
* 3 = stairs down
* 4 = stairs up

Because the accelerometers are sampled at high frequency, the labels in train_labels.csv are only provided for every 10th observation in train_time_series.csv

![](https://github.com/vanessadlafp/HarvardX_PH526x/blob/master/Images/time-series.png)

## Analysis

### Check for Imbalanced class
![](https://github.com/vanessadlafp/HarvardX_PH526x/blob/master/Images/class_imbalance.png)

The data isn't well balanced, having considerably more data points for walking (2) activity, followed by stairs down (3). This might be related with the lifestyle of the subjects of study, however, should be considered when builiding the model to avoid any bias towards the more dominant classes. 

### Variable analysis

* Moving and stationnary activities behave diferently, where the magnitude of of linear acceleration of the latter remains in ranges lower than those of stationary activities.

![](https://github.com/vanessadlafp/HarvardX_PH526x/blob/master/Images/moving_vs_stationnary.png)

![](https://github.com/vanessadlafp/HarvardX_PH526x/blob/master/Images/moving_vs_stationnary_zoomed_in.png)

* Mean of magnitude of acceleration:

![](https://github.com/vanessadlafp/HarvardX_PH526x/blob/master/Images/acceleration_per_component.png)

## Featurization

* Calculation of the magnitude of each vector from its components using the using the Pythagorean Theorem, included to dataset as column "m".
* Parsing of 'UTC Time' column from string to timestamp.
* Sampling of time and frequency for time series signals and labels.
* Applied Frequency Transformation Functions (Fast Fourier Transform (FFT), Power Spectral Density (PSD) and Auto-correlation) to transform the signals from the time-domain to  the frequency-domain and extract features from them (frequencies at which oscillations occur and their corresponding amplitudes).  Following along with Ahmet Taspinar article referenced above. 


![](https://github.com/vanessadlafp/HarvardX_PH526x/blob/master/Images/feature_extraction.png) 

## Resampling for class imbalance

Considering that the most dominant class contains over 50% of the total data, Imbalanced-learn's Oversample Adaptive Synthetic (ADASYN) algorith was used.

This algorithm generates different number of samples depending on an estimate of the local distribution of the class to be oversampled.

* Original dataset shape Counter({2: 213, 3: 88, 4: 47, 1: 27})
* Resampled dataset shape Counter({3: 231, 2: 213, 1: 212, 4: 209})

## Models

* Split the train data into train and validation sets with a validation size of 20%
* Scikit-learn is used for all the 5 algorithms listed below to select the best one for hyperparameter tuning, taking into condideration the train time:

![](https://github.com/vanessadlafp/HarvardX_PH526x/blob/master/Images/models_table.png)

Hyperparameters of Random Forest model are tuned by grid search CV, even though it was the second best performer, it was able to achieve a high accuracy on the validation data in considerably less time than the Gradient Boost algorithm. 


## Productionization

Building of Flask API endpoint that was hosted on a local webserver by following along with the TDS and Ken Jee's tutorials, referenced in the section above. The API endpoint takes in a request with a vector with frequency and amplitude features (taken from the components x,y,z and magnitude of measurements of linear acceleration) and predicts the type of physical activity (e.g., walking, climbing stairs).