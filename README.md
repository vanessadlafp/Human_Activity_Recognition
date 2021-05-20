# HarvardX PH526x- Using Python for Research 
## Capstone Project: Project Overview

* Created a tool that XXX  (ERROR ~ $ 11K) to XXX.
* Engineered features from XXX on python, excel, aws, and spark.
* Optimized (Linear, Lasso, and Random Forest==BUSCAR EQUIVALENTES PARA CLASSIFICATION) Regressors using GridsearchCV(VER SI SIRVE PARA CLASSIFIERS) to reach the best model.
* Built a client facing API using flask

## Project Instructions:
#### Introduction

In this final project,we'll attempt to predict the type of physical activity (e.g., walking, climbing stairs) from tri-axial smartphone accelerometer data. Smartphone accelerometers are very precise, and different physical activities give rise to different patterns of acceleration.

#### Input Data
The input data used for training in this project consists of two files. The first file, train_time_series.csv
, contains the raw accelerometer data, which has been collected using the Beiwe research platform, and it has the following format:
timestamp, UTC time, accuracy, x, y, z
You can use the timestamp column as your time variable; you'll also need the last three columns, here labeled x
, y
, and z
, which correspond to measurements of linear acceleration along each of the three orthogonal axes.

The second file, train_labels.csv
, contains the activity labels, and you'll be using these labels to train your model. Different activities have been numbered with integers. We use the following encoding: 1 = standing, 2 = walking, 3 = stairs down, 4 = stairs up. Because the accelerometers are sampled at high frequency, the labels in train_labels.csv
 are only provided for every 10th observation in train_time_series.csv
.

#### Activity Classification

Your goal is to classify different physical activities as accurately as possible. To test your code, you're also provided a file called test_time_series.csv
, and at the end of the project you're asked to provide the activity labels predicted by your code for this test data set. Note that in both cases, for training and testing, the input file consists of a single (3-dimensional) time series. To test the accuracy of your code, you'll be asked to upload your predictions as a CSV file. This file called test_labels.csv is provided to you, but it only contains the time stamps needed for prediction; you'll need to augment this file by adding the corresponding class predictions (1,2,3,4).

## Code and Resources Used
Python Version: 3.7
Packages: pandas, numpy, sklearn, matplotlib, seaborn, flask, json, pickle
For Web Framework Requirements: pip install -r requirements.txt
X Github: 
X Article: 
Flask Productionization: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## EDA
(CAMBIAR PERO MODELO)
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables.)

## Model Building
(CAMBIAR PERO MODELO)
First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.

I tried three different models:

Multiple Linear Regression – Baseline for the model
Lasso Regression – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
Random Forest – Again, with the sparsity associated with the data, I thought that this would be a good fit.

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets.

Random Forest : MAE = 11.22
Linear Regression: MAE = 18.86
Ridge Regression: MAE = 19.67

## Productionization

In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary.