"""
plotFeatureSelection.py

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
#import numpy as np


def get_datos():
    file = 'datos.csv'
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion ver2"
    path = path.replace('\\', '/')
    path = os.path.join(path, file)
    
    df = pd.read_csv(path, sep=';')
    X = df.iloc[:, 4:].values
    y = df['dpi'].values
    n = X.shape[1]

    return X, y, n

def plot_lasso_features(estimador):
    X, y, n = get_datos()
    estimador.fit(X,y)
    y_predict = estimador.predict(X)
    score = mean_absolute_error(y_true=y, y_pred=y_predict)
    score = round(score, 3)
    print('mean_absolute_error: {}'.format(score))

    pesos = estimador.coef_
    pesos = pesos.reshape(-1)

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    x = [i for i in range(n)]
    # marker='o', markersize=1, linestyle='None',
    ax.plot(x, pesos, color='blue', marker='o', markersize=3, linestyle='None',label='peso coef')

    ax.legend()
    plt.show()



