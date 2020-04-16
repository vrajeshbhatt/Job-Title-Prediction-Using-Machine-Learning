# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 02:31:03 2020

@author: Vrajesh
"""
from flask import Flask, request,  render_template, Response
import numpy as np
from keras.models import load_model
from keras import backend as K
from flask import jsonify
from keras.models import model_from_json


app=Flask(__name__)


@app.route("/")
def home():

   
    return render_template('quiz.html')


@app.route("/predict",methods=['POST'])
def predict():

    job_title={0:'Database Administator',1:'Embedded System Engineer',2:'Appliction Developer',3:'Ethical Hacker',
           4:'Computer Hardware & Software Implementation & Maintence',5:'Game Development',6:'Web Designer',
           7:'Computer programmer',8:'UI/UX Designer',9:'Software Developer',10:'Data Warehouse Analyst',
           11:'Software Engineer',12:'Networking Speaciallist',13:'E-Commerce Speacialist',14:'Project Manager'}


    int_features=[int(x) for x in request.form.values()]
    final_features=np.array(int_features)
    final_features.resize(1,50)
#    prediction=generate_prediction(final_features)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    pred=loaded_model.predict_classes(final_features)
    name=pred[0]
    return render_template('result.html',prediction_text='Most Suitable job for you is {}'.format(job_title[name]))
     

if __name__=='__main__':
    app.run()