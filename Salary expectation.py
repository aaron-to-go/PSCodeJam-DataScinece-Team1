#-------------------------------------------------------------------------------
# Name:        Salary expectation
# Purpose:     Simple neural network, adapted from Kanrad' teachings, for DS Team 1 - Code jam
# Author:      emanu
# Created:     15/02/2022
# Copyright:   (c) emanu 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

"""to be used in Google colab - note that there you run the code step
by step, not entirely at once"""

#step 1 - import pandas and dataset

import pandas as pd
dataset = pd.read_csv("name.csv")

"""the name of the cvs should be the name of the actual file"""

#step 2 - we want to predict salary (y) based on n feauters (x)

x = dataset.drop(columns=["column_name (1=a, 0=b)"])

"""here I simplified the analisys in a (above mean) and b (below mean);
the column name should be the same of the cvs file that you uploaded, and it's
removed because is our target"""

#step 3 -

y = dataset("colum_name(1=a, 0=b)")

#step 4 - splitting the data between training and testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

"""0.1 stands for 10% of the values dedicated to testing, but you can chose
to increment or decrease it at your will"""

#step 5 - implement the neural network using keras

import tensorflow as tf

model = tf.keras.models.Sequential()

#step 6 - layers

model.add(tf.keras.layers.Dense(42, input_shape = x_train.shape, activation= 'sigmoid'))
model.add(tf.keras.layers.Dense(42, activation= 'sigmoid'))
model.add(tf.keras.layers.Dense(1, activation= 'sigmoid'))

"""dense is a type of neuron, the number of neurons and layers can vary, I use
the number of columns of the dataset to verify an idea, but it's up to you...
the last one is just one becouse of the target"""

#step 7 - module

module.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#step 8 - feed data

model.fit(x_train, y_train, epochs=100)

"""epochs equals number of total iterations...is up to you. NOTE that when you
run this in colab, it will takes time to complete, more neurons more layers
more epochs = more time!!!"""

#step 9 - evaluation

model.evaluate(x_test, y_test)


