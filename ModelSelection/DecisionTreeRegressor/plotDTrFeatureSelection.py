"""
plotDTrFeatureSelection.py

"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def get_datos():
    file = 'datos.csv'
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion ver2"
    path = path.replace('\\', '/')
    path = os.path.join(path, file)

    df = pd.read_csv(path, sep=';')
    y = df['dpi'].values
    X = df.iloc[:, 4:].values
    return X, y

def plotFeatureSelection(estimador):
    X, y = get_datos()
    estimador.fit(X, y)
    y_predict = estimador.predict(X)
    x = mean_absolute_error(y_pred=y_predict, y_true=y)
    x = round(x, 3)
    print('mean_absolute_error: {}'.format(x))

    importances = estimador.feature_importances_

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    x = [i for i in range(len(importances))]
    ax.plot(x, importances, color='blue', marker='o', markersize=3, linestyle='None',label=r"% importancia features")

    ax.legend()
    plt.show()

