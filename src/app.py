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
from model.model import create_model
from threading import Thread

app = Flask(__name__)
api = Api(app)
CORS(app)

# Path: init.py
# load the model
create_model()
model = joblib.load('model/model.pkl')

# create main page that give the instructions about how to use the API


@app.route('/')
def main():
    return '''<h1>Wine Quality Prediction</h1>
<p>A prototype API for predicting wine quality.</p>'''


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
    data = pd.read_csv('model/data/winequality.csv')
    if head:
        data = data.head(10)
    return {"data": data.to_string()}

# endpoint that gives the performance of the model


@app.route('/performance', methods=['GET'])
def performance():
    # load the metrics of the model from metrics.txt
    with open('model/metrics.txt', 'r') as f:
        metrics = f.read()
        metrics = metrics.split('\n')
    res = []
    for m in metrics:
        metric = m.split(':')
        res.append((metric[0],metric[1]))
    # return data
    return jsonify(results=res)


@app.route('/predict', methods=['PUT'])
def predict():
    # get the data from the POST request.
    # The data is in the form of a string and does not have the quality in it
    data = request.get_json()['data'].replace(" ", "").split(',')
    data_df = pd.DataFrame([data], columns=['fixed acidity', 
    'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])

    # predictions
    result = model.predict(data_df)

    return jsonify({'results': str(result[0])})

# add an endpoint to retrain the model with new data
@ app.route('/retrain', methods=['PUT'])
def retrain():
    # get the data
    data = request.get_json(force=True)['data'].split('\n')
    # remove all empty lines
    data = [d for d in data if d != '']
    # separate the columns from the data
    columns = data[0].split(',')
    data = [ d.split(',') for d in data[1:]]

    # convert data into dataframe
    data_df = pd.DataFrame(data, columns=columns)
    # separate input features and target
    X = data_df.iloc[: , : -1]
    y = data_df.iloc[:, -1]
    # retrain the model
    global model
    model = model.fit(X, y)
    # create a path to the model
    modelpath = os.path.join('model/model.pkl')
    # save the model
    joblib.dump(model, modelpath)
    # return a message
    return jsonify('Model successfully retrained.')

if __name__ == '__main__':
    app.run(port='5002')
