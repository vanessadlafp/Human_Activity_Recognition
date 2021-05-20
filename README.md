# HarvardX PH526x- Using Python for Research 
## Capstone Project: Project Overview

### Instructions describing the project:
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

