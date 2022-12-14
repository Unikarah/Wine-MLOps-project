# create model that classifies the data in data/winequality.csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import numpy as np

def create_model(display_metrics=False):
    # import the data from data/winequality.csv
    data = pd.read_csv('data/winequality.csv')

    # split the data into training and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # separate the features from the labels
    train_features = train.drop('quality', axis=1)
    train_labels = train['quality']

    test_features = test.drop('quality', axis=1)
    test_labels = test['quality']

    # save test data in file test_wine.csv
    test.to_csv('data/test_wine.csv', index=False)

    # create a model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    # train the model
    model.fit(train_features, train_labels)

    # evaluate the model
    predictions = model.predict(test_features)

    if display_metrics:
        # print the metrics
        print('Accuracy:', metrics.accuracy_score(test_labels, predictions))
        print('Precision:', metrics.precision_score(test_labels, predictions, average='weighted'))
        print('Recall:', metrics.recall_score(test_labels, predictions, average='weighted'))
        print('F1 score:', metrics.f1_score(test_labels, predictions, average='weighted'))

    # save the model in file model.pkl
    pickle.dump(model, open('model.pkl', 'wb'))