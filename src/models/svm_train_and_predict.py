import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

essays = pd.read_csv('../../data/processed/essays_with_topic_scores.csv', index_col=0)

# Set the essay id as the index of the dataframe
essays.set_index('essay_id', inplace=True)

# scale the length column so that all features are between 0 - 1
mm_scaler = preprocessing.MinMaxScaler()
essays['length'] = mm_scaler.fit_transform(essays[['length']])

# separate your features and label
X = essays.drop(['domain1_score'], axis=1)
y = essays['domain1_score']

# split data set into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=34)


# Instantiate the GridSearchCV object and run the search
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)

# instantiate a new SVC with those hypertuned paramaters
svm = SVC(kernel='rbf', random_state=12, gamma=0.1, C=10)

# fit the data to the model and predict
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# print the accuracy score
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))