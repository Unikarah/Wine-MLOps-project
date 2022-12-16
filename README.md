# MLOPS

## Authors
Sarah Gutierez \
Adrien Houpert

## About the model

The model implementation can be found in the file `src/model/model.py`. It uses the 
[Red Wine Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?select=winequality-red.csv) from Kaggle.

## To run the project
The hole project:

```bash
docker-compose up --build
```

Run only the API:

```bash
python api.py
```

## To test the API

You have multiple way to test the API.

You can either do it by hand in your terminal using curl and the adress : 'http://127.0.0.1:5002'.

Or you can use our wonderful frontend and just click on the different parts!
