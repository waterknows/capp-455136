#Machine Learning Pipeline
The goal of this project is to build a task-oriented toolkit on machine learning problems. 
The toolkit makes use of the powerful packages such as sklearn, pandas, numpy in Python and build a function for each machine learning task.

This toolkit normalizes the standard machine learning process into the following six steps: 

1) Preparing Data

2) Exploring Data

3) Generating Features

4) Running Models

5) Evaluating Models

To use this package in Python:

1) Clone the repository into your computer 

2) Set up your local path in python: import os, os.chdir("local path")

3) Import the package: from mlpipe import *

4) Call functions from each submodule

For example, to fill in missing values in your database with KNN method, you just need to call the function proc.KNN:
proc.KNN(dataframe name, the varaible with missing values; features used to predict the missing values; number of neighbors k)
