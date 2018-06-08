import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def main():
    df = data_intake()
    df_manipulation = data_manipulation(df)

    # adding manipulated data to different sklearn algorithms
    linreg_score = linear_regression(df_manipulation)
    knn_score = knn(df_manipulation)
    logreg_score = logistic_regression(df_manipulation)

def data_intake():
    # intaking data
    df = pd.read_csv('Iris.csv')
    # changing the species to numbers so that it can be manipulated
    df['Species'] = np.where(df['Species'] == 'Iris-setosa', 0, np.where(df['Species'] == 'Iris-versicolor', 1, 2))
    # dropping columns that aren't relevant
    df.drop('Id', axis=1)
    return df

def data_manipulation(df):
    # Relevant values are = X
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    # Y is what we're trying to find
    Y = df['Species']
    # splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state=52)
    return df

def linear_regression(df):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    linreg_predict = linreg.predict(X_test)
    linreg_score = linreg.score(X_train, y_train)
    return linreg_score

def knn(df):
    KNN = KNeighborsClassifier()
    KNN.fit(X_train, y_train)
    KNN_predict = KNN.predict(X_test)
    KNN_score = KNN.score(X_train, y_train)
    return KNN_score

def logistic_regression(df):
    linreg = LogisticRegression()
    linreg.fit(X_train, y_train)
    linreg_predict = linreg.predict(X_test)
    linreg_score = linreg.score(X_train, y_train)
    return linreg_score