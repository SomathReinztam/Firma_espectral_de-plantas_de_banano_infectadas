"""
data_augmentation1.py

"""
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

os.environ['LOKY_MAX_CPU_COUNT'] = '5'  # Reemplaza 4 con el número de núcleos que tienes

file = 'newdatos.csv'
path = r"C:\Users\Acer\Documents\python\Espectro Plantas Banano\Nuevos Datos\Procesaminto nuevos datos"
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')

X = df.iloc[:, 3:].values
y = df['Sana'].values

n = 2000
smote = SMOTE(sampling_strategy={-1:n, 1:n}, k_neighbors=5, random_state=42)

X, y = smote.fit_resample(X, y)

print(f'hay {np.sum((y == -1))}, {np.sum((y == 0))}, {np.sum((y == 1))} en la clase -1, 0, 1')
print()


def plot_datos_PCA2d_augmentation(X, y):
    pca = PCA(n_components=2)
    A = pca.fit_transform(X)
    df = pd.DataFrame(A, columns=['eje_x', 'eje_y'])
    df['Sana'] = y

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    Sana = [0, 1, -1]
    Colores = ['#FF7400', '#3EE600', '#0B29C9']
    Label = ['Fusarium', 'Sana', 'HyS']
    for i in range(len(Sana)):
        filtro = (df['Sana'] == Sana[i])
        df_filtro = df[filtro]
        x = df_filtro['eje_x'].values
        y = df_filtro['eje_y'].values
        ax.scatter(x, y, color=Colores[i], marker='o', label=Label[i])
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def plot_datos_LDA2d_augmentation(X, y):
    pca = LinearDiscriminantAnalysis(n_components=2)
    A = pca.fit_transform(X, y)
    df = pd.DataFrame(A, columns=['eje_x', 'eje_y'])
    df['Sana'] = y

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    Sana = [0, 1, -1]
    Colores = ['#FF7400', '#3EE600', '#0B29C9']
    Label = ['Fusarium', 'Sana', 'HyS']
    for i in range(len(Sana)):
        filtro = (df['Sana'] == Sana[i])
        df_filtro = df[filtro]
        x = df_filtro['eje_x'].values
        y = df_filtro['eje_y'].values
        ax.scatter(x, y, color=Colores[i], marker='o', label=Label[i])
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


#plot_datos_PCA2d_augmentation(X, y)

plot_datos_LDA2d_augmentation(X, y)

"""

import os

nucleos_logicos = os.cpu_count()
print(f"Número de núcleos lógicos (con hyperthreading): {nucleos_logicos}")


"""