# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:26:05 2020

@author: Vrajesh
"""

import pandas as pd
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from numpy import loadtxt
from keras.models import model_from_json
dataset=pd.read_csv('dataset.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y=labelencoder.fit_transform(y)
from keras.utils import to_categorical

y=to_categorical(y)

from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) 

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout

model=Sequential()
model.add(Dense(200,input_dim=50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(15,activation='softmax'))

model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=5,epochs=30)
score=model.evaluate(x_test,y_test)
score
predictions = model.predict_classes(x_test)
# summarize the first 5 cases
for i in range(20):
	print(predictions[i]," => ",y_test[i])


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score_model = loaded_model.evaluate(x_test, y_test, verbose=0)

print(score_model)

