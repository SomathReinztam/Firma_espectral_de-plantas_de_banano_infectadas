"""
plotFeatureSelection.py

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np


def get_datos():
    file = 'datos.csv'
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion ver2"
    path = path.replace('\\', '/')
    path = os.path.join(path, file)
    
    df = pd.read_csv(path, sep=';')
    X = df.iloc[:, 4:].values
    y = df['Sana'].values
    n = X.shape[1]

    return X, y, n

def plot_logreg_features(estimador):
    X, y, n = get_datos()
    estimador.fit(X,y)
    score = estimador.score(X, y)
    score = round(score, 3)
    print('score: {}'.format(score))

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





