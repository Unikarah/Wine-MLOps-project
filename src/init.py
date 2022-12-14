from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonpify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os
from model import create_model
from threading import Thread

app = Flask(__name__)
api = Api(app)
CORS(app)

# Path: init.py
# load the model
create_model()
model = joblib.load('model.pkl')

# create main page that give the instructions about how to use the API


@app.route('/')
def main():
    return '''<h1>Wine Quality Prediction</h1>
<p>A prototype API for predicting wine quality.</p>'''

# get all the possible endpoints


@app.route('/endpoints')
def endpoints():
    endpoints = ['/winequality', '/winequality/performance',
                 '/winequality/predict', '/winequality/retrain']
    return jsonify(endpoints)

@app.route('/model')
def info():
    # get the model informations
    model_info = model.get_params()
    # return data
    return jsonify(model_info=model_info)

# endpoint that shows the base data of the model
@app.route('/data', methods=['GET'])
def datahtml(head=True):
    # load the data
    data = pd.read_csv('data/winequality.csv')
    if head:
        data = data.head(10)
    return {"data" : data.to_string() }

# endpoint that gives the performance of the model
@app.route('/performance', methods=['GET'])
def performance(self):
    # get the data
    data = request.get_json(force=True)
    # convert data into dataframe
    data_df = pd.DataFrame(data)
    # predictions
    result = model.predict(data_df)
    # send back to browser
    output = {'results': int(result[0])}
    # return data
    return jsonpify(results=output)

@app.route('/predict', methods=['PUT'] )
def predict():    
    print("helloooo")
    # get the data from the POST request.
    # The data is in the form of a string and does not have the quality in it
    print(request.get_json()['data'].split(','))
    data = request.get_json()['data'].split(',')
    # data_df = pd.DataFrame([data])
    #print("helloooo", data_df)
    # predictions
    #result = model.predict(data_df)

    return jsonify({'results': 1 }) # result[0]})

# add an endpoint to retrain the model with new data
@app.route('/retrain', methods=['PUT'])
def retrain():
    # get the data
    request_json = request.get_json(force=True)
    print(request_json)
    if 'data' in request_json and request_json['data'] != '':
        # get the data from the PUT request
        data = [ x.split(",") for x in request.get_json(force=True)['data'].split("\n") ]
    elif 'file' in request_json and request_json['file'] != '':
        if not os.path.exists(request.get_json(force=True)['file']):
            return jsonify('File does not exist.')
        
        # get the data from the PUT request
        data = pd.read_csv(request.get_json(force=True)['file'])
        data = data.values.tolist()
    else:
        return jsonify('No data provided.')

    # convert data into dataframe
    data_df = pd.DataFrame(data)
    # separate input features and target
    X = data_df.iloc[:, :-1]
    y = data_df.iloc[:, -1]
    # retrain the model
    global model
    model = model.fit(X, y)
    # create a path to the model
    modelpath = os.path.join('model.pkl')
    # save the model
    joblib.dump(model, modelpath)
    # return a message
    return jsonify('Model successfully retrained.')

if __name__ == '__main__':
    app.run(debug=True)
    app.run(port='8080') 