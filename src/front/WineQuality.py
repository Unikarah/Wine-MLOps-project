import streamlit as st
from front_utils import *

# BASE OF THE PAGE

# create a simple API frontend using streamlit
st.title(':wine_glass:' + '_Red Wine Quality Prediction_ ' + ':wine_glass:')
st.header('An API for predicting wine quality.')
st.markdown(
    f'The API is available at http://127.0.0.1:5002 and the code is available on Github\
        [here](https://github.com/Unikarah/MLOPS)')
st.markdown("Our objective is to predict the quality of a wine based on its\
     chemical properties. As it can be seen in the data sample bellow, the quality \
     of a wine is a number between 0 and 10. The quality is determined by\
    a panel of experts. The dataset contains 1599 wines with 12 \
    features and the quality of each wine. The features are the \
    following: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, \
        free sulfur dioxide, total sulfur dioxide, density, pH, sulphates and alcohol.")


get_data_info()

# change the size of the page
st.header('The model')
st.subheader("The dataset :floppy_disk:")
st.markdown(
    'The model uses the \
        [Red Wine Quality](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) dataset on Kaggle.')
st.markdown("The dataset is available in the data folder of the repository.")
st.markdown("The dataset is split into a training set and a test set. The training set is used to train the model and the test set is used to evaluate the performance of the model.")
st.markdown("The dataset is also available in the API at /data")
st.markdown("Each column of the dataset is described bellow:")
st.markdown("- **fixed acidity**: most acids involved with wine or fixed or nonvolatile (do not evaporate readily)")
st.markdown("- **volatile acidity**: the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste")
st.markdown("- **citric acid**: found in small quantities, citric acid can add 'freshness' and flavor to wines")
st.markdown("- **residual sugar**: the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet")
st.markdown("- **chlorides**: the amount of salt in the wine")
st.markdown("- **free sulfur dioxide**: the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine")
st.markdown("- **total sulfur dioxide**: amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine")
st.markdown("- **density**: the density of water is close to that of water depending on the percent alcohol and sugar content")
st.markdown("- **pH**: describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic)")
st.markdown("- **sulphates**: a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant")
st.markdown("- **alcohol**: the percent alcohol content of the wine")
st.markdown("- **quality**: output variable (based on sensory data, score between 0 and 10)")

st.subheader("The model :brain:")
st.markdown("The model is a Gradient Bososting Classifier. The model is trained on the data and the performance is evaluated using the accuracy, recision, the recall and the F1 score. The model is then saved and can be used to predict the quality of a wine.")
st.markdown("In the following sections you can see the model informations and\
    the performance of the model.")
with st.expander("The model informations :information_source:"):
    get_model_info()

with st.expander("The performance of the model :chart_with_upwards_trend:"):
    get_performance()

st.header("Interact with the model")
st.markdown("In the following sections you can interact with the model. You can\
    predict the quality of a wine based on its characteristics or retrain the model")
with st.expander("Predict the quality of a wine :wine_glass:"):
    predict_quality()

with st.expander("Retrain the model :repeat:"):
    uploaded_file = st.file_uploader(
        "Choose a CSV with the new wine data", type="csv")
    if uploaded_file is not None:
        # store the csv text in a variable
        csv_text = uploaded_file.read().decode('utf-8')
        retrain_model(csv_text)

st.header("The API")
st.markdown("Here are the endpoints of the API:")
st.markdown("- The model informations are available at /model")
st.markdown("- The data used to train the model are available at /data")
st.markdown("- The performance of the model are available at /performance")
st.markdown("- The prediction of the model are available at /predict")
st.markdown("- The model can be retrained at /retrain")

st.markdown('''
    <style> [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    } </style>
    ''', unsafe_allow_html=True)

st.header("Authors")
st.markdown("This project was made by Sarah Gutierez and Adrien Houpert for the MLOPS course at EPITA.")
# END OF THE PAGE
