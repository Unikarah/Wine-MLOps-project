import streamlit as st
import requests
import pandas as pd

url = "http://127.0.0.1:8080"

# get the model informations
def get_model_info():
    st.write('The model informations are:')
    st.json(requests.get(f"{url}/model").json())

# show the base data of the model
def get_data_info():
    st.write('Here is a sample of the base data for the model:')
    result = requests.get(f"{url}/data").json()['data'].split('\n')
    result = pd.DataFrame(result[1:], columns=[result[0]])
    st.dataframe(result)


# give a file to upload
def retrain_model(uploaded_file):
    data = {'data': uploaded_file }
    res = requests.put(f"{url}/retrain", json=data)
    if res.status_code != 200:
        st.write("Error, the file is not valid!")
    else:
        st.write('The model has been retrained with the new data!')

# Predicts the quality of a given wine characteristics
def predict_quality():
    # get the data from the POST request. The data is in the form of a string
    # and does not have the quality in it
    st.write('Please insert the wine characteristics:')
    col1, col2, col3 = st.columns(3)
    with col1:
        fixed_acidity = st.text_input('Insert the fixed acidity:')
    with col2:
        volatile_acidity = st.text_input('Insert the volatile acidity:')
    with col3:
        citric_acid = st.text_input('Insert the citric acid rate:')
    col4, col5, col6 = st.columns(3)
    with col4:
        residual_sugar = st.text_input('Insert the residual sugar:')
    with col5:
        chlorides = st.text_input('Insert tjhe chlorides:')
    with col6:
        free_sulfur_dioxide = st.text_input('Insert the free sulfur dioxide:')
    col7, col8, col9 = st.columns(3)
    with col7:
        total_sulfur_dioxide = st.text_input('Insert the total sulfur dioxide:')
    with col8:
        density = st.text_input('Insert the density:')
    with col9:
        pH = st.text_input('Insert the pH:')
    col10, col11 = st.columns(2)
    with col10:
        sulphates = st.text_input('Insert the sulphates:')
    with col11:
        alcohol = st.text_input('Insert the alcohol:')

    data = f"{fixed_acidity}, {volatile_acidity}, {citric_acid}, {residual_sugar}, \
        {chlorides}, {free_sulfur_dioxide}, {total_sulfur_dioxide}, {density}, {pH}, {sulphates}, {alcohol}"

    # if the data is inserted send it to the API
    if fixed_acidity != '' and volatile_acidity != '' and citric_acid != '' and residual_sugar != '' and chlorides != '' \
            and free_sulfur_dioxide != '' and total_sulfur_dioxide != '' and density != '' and pH != '' and sulphates != '' and alcohol != '':
        data = {'data': data}
        res = requests.put(f"{url}/predict", json=data)
        st.write('The models prediction about the wine quality is:')
        res = res.json()['results']
    
        result_str = ''
        for i in range(int(res)):
            result_str += ':heart:'
        for i in range(10-int(res)):
            result_str += ':black_heart:'

        st.write(result_str)   

# get the performance of the model
def get_performance():
    st.write('The performances of the model are:')
    results = requests.get(f"{url}/performance").json()['results']
    st.write("Here are the performances of the model:")
    for (name, val) in results:
        st.markdown(f"- {name}: {val}")

    st.markdown('''
    <style> [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    } </style>
    ''', unsafe_allow_html=True)
