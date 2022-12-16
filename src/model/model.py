# create model that classifies the data in data/winequality.csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import numpy as np

def create_model():
    # import the data from data/winequality.csv
    data = pd.read_csv('model/data/winequality.csv')

    # split the data into training and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # separate the features from the labels
    train_features = train.drop('quality', axis=1)
    train_labels = train['quality']

    test_features = test.drop('quality', axis=1)
    test_labels = test['quality']

    # save test data in file test_wine.csv
    test.to_csv('model/data/test_wine.csv', index=False)

    # create a model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    # train the model
    model.fit(train_features, train_labels)

    # evaluate the model
    predictions = model.predict(test_features)

    # save the metrics in file metrics.txt
    with open('model/metrics.txt', 'w') as f:
        f.write('For the accuracy:' + str(metrics.accuracy_score(test_labels, predictions)) + '\n')
        f.write('For the recision: ' + str(metrics.precision_score(test_labels, predictions, average='weighted')) + '\n')
        f.write('For the recall:' + str(metrics.recall_score(test_labels, predictions, average='weighted')) + '\n')
        f.write('For the F1 score:' + str(metrics.f1_score(test_labels, predictions, average='weighted')))

    # save the model in file model.pkl
    pickle.dump(model, open('model/model.pkl', 'wb'))
