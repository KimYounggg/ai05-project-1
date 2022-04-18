# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:24:00 2022

@author: Kim Young
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Read the csv data
bc_data = pd.read_csv(r"C:\Users\Kim Young\Desktop\SHRDC\Deep Learning\TensorFlow Deep Learning\Datasets\data.csv")

#%%
#Drop columns which are not useful
bc_data = bc_data.drop(['id', 'Unnamed: 32'], axis = 1)

#Split into features and labels
bc_features = bc_data.copy()
bc_labels = bc_features.pop('diagnosis')

#%%
#One hot encode label
bc_labels_OH = pd.get_dummies(bc_labels)

#%%
#Split features and labels into train-validation-test sets
x_inter, x_eval, y_inter, y_eval = train_test_split(bc_features, 
                                                    bc_labels_OH, 
                                                    test_size = 0.2, 
                                                    random_state = 12345)

x_train, x_test, y_train, y_test = train_test_split(x_inter, 
                                                    y_inter, 
                                                    test_size = 0.4, 
                                                    random_state = 12345)

#%%
#Normalize the features
ss = StandardScaler()
ss.fit(x_train)

x_eval = ss.transform(x_eval)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

#%%
from tensorflow.keras.layers import InputLayer, Dense, Dropout

#Create feedforward neural network
inputs = x_train.shape[-1]
nClass = y_train.shape[-1]

model = tf.keras.Sequential()

model.add(InputLayer(input_shape = inputs))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(nClass, activation = 'softmax'))

#Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#%%
#Train model
log_path = r'C:\\Users\\Kim Young\\Desktop\\SHRDC\\Deep Learning\\TensorFlow Deep Learning\\Tensorboard\\logs'
tb = tf.keras.callbacks.TensorBoard(log_dir = log_path)
es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)

history = model.fit(x_train, y_train, validation_data = (x_eval, y_eval), batch_size = 32,epochs = 100, callbacks = [es,tb])

#%%
#Evaluate the model with test data
result = model.evaluate(x_test, y_test, batch_size = 32)

print(f"Test loss = {result[0]}")
print(f"Test accuracy = {result[1]}")

#%%
#Predictions
pred = model.predict(x_test)
pred = np.argmax(pred, axis = 1)
print(pred)