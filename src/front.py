import streamlit as st
import requests
import pandas as pd
import joblib
from model import create_model


# BASE OF THE PAGE

url = "http://127.0.0.1:8080"
# create a simple API frontend using streamlit
st.title('_Wine Quality Prediction_ ' + ':wine_glass:')
st.header('A prototype API for predicting wine quality.')
st.subheader('**This is a simple API frontend using streamlit.')
st.markdown(
    'The API is hosted on Heroku and the code is available on Github [here](git@github.com:Unikarah/MLOPS.git)')
st.write(f'The API is available at {url}')

# LAYOUT SETUP

# change the size of the page


# FUNCTIONS FOR THE ENDPOINTS !

def get_all_endpoints():
    # get all the possible endpoints
    st.write('If youThe possible endpoints are:')

    # create list of endpoints
    st.markdown("- '/' ")
    st.markdown("- '/winequality'")
    st.markdown("- '/winequality/performance'")
    st.markdown("- '/winequality/predict'")
    st.markdown("- '/winequality/retrain'")

    st.markdown('''
    <style> [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    } </style>
    ''', unsafe_allow_html=True)

# get the model informations


def get_model_info():
    st.write('The model informations are:')
    st.json(requests.get(f"{url}/model").json())

# show the base data of the model


def get_data_info():
    st.write('The base data of the model is:')
    result = requests.get(f"{url}/data").json()['data'].split('\n')
    result = pd.DataFrame(result[1:], columns=[result[0]])
    st.dataframe(result)


# give a file to upload
def retrain_data():
    uploaded_file = st.file_uploader(
        "Choose a CSV with the new wine data", type="csv")

    # if the file is uploaded send it to the API
    if uploaded_file is not None:

        data = {'data': pd.read_csv(uploaded_file).to_string()}

        res = requests.put(f"{url}/retrain", json=data)
        if res.status_code != 200:
            st.write("Error, the file was not valid!")
        else:
            st.write('The model has been retrained with the new data!')

# Predicts the quality of a given wine characteristics
def predict_quality():
    # get the data from the POST request. The data is in the form of a string
    # and does not have the quality in it
    st.write('Insert the data to predict the quality of the wine')
    fixed_acidity = st.text_input('Insert the fixed acidity')
    volatile_acidity = st.text_input('Insert the volatile acidity')
    citric_acid = st.text_input('Insert the citric acid rate')
    residual_sugar = st.text_input('Insert the residual sugar')
    chlorides = st.text_input('Insert tjhe chlorides')
    free_sulfur_dioxide = st.text_input('Insert the free sulfur dioxide')
    total_sulfur_dioxide = st.text_input('Insert the total sulfur dioxide')
    density = st.text_input('Insert the density')
    pH = st.text_input('Insert the pH')
    sulphates = st.text_input('Insert the sulphates')
    alcohol = st.text_input('Insert the alcohol')

    data = f"{fixed_acidity}, {volatile_acidity}, {citric_acid}, {residual_sugar}, \
        {chlorides}, {free_sulfur_dioxide}, {total_sulfur_dioxide}, {density}, {pH}, {sulphates}, {alcohol}"

    # if the data is inserted send it to the API
    if fixed_acidity is not None and volatile_acidity is not None and citric_acid is not None and residual_sugar is not None and chlorides is not None \
            and free_sulfur_dioxide is not None and total_sulfur_dioxide is not None and density is not None and pH is not None and sulphates is not None and alcohol is not None:
        data = {'data': data}
        res = requests.put(f"{url}/predict", json=data)
        st.write('The predictions are:')
