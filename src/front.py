import streamlit as st
import pandas as pd
import joblib
from model import create_model


create_model()
model = joblib.load('model.pkl')

# create a simple API frontend using streamlit
st.title('Wine Quality Prediction')
st.write('A prototype API for predicting wine quality.')
st.write('This is a simple API frontend using streamlit.')
st.write('The API is hosted on Heroku and the code is available on Github.')
st.write('The API is available at https://wine-quality-prediction.herokuapp.com/')
st.write('The Github repository is available at ')

# get all the possible endpoints
st.write('The possible endpoints are:')
st.write('/winequality')
st.write('/winequality/performance')
st.write('/winequality/predict')    
st.write('/winequality/retrain')

# get the model informations
st.write('The model informations are:')
st.write(model.get_params())

# show the base data of the model
st.write('The base data of the model is:')
st.write(pd.read_csv('data/winequality.csv').head())

# give a file to upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if the file is uploaded send it to the API
if uploaded_file is not None:
    # read the file
    data = pd.read_csv(uploaded_file)
    # get the predictions
    result = model.predict(data)
    # show the predictions
    st.write('The predictions are:')
    st.write(result)

# function that calls the API endpoint /predict