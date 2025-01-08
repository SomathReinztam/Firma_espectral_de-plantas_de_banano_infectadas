"""
4_Model_DataLoader.py

"""

from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
#import numpy as np
from CNNmodel import EspectroDataset, SanaModel
from torch.nn import BCELoss


def get_datos():
    file = 'datos.csv'
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion ver2"
    path.replace('\\', '/')
    path = os.path.join(path, file)
    df = pd.read_csv(path, sep=';')
    X = df.iloc[:, 4:].values
    y = df['Sana'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=12
    )
    return X_train, y_train

X_train, y_train = get_datos()

dataset = EspectroDataset(inputs=X_train, labels=y_train)
train_dl = DataLoader(dataset=dataset, batch_size=8, shuffle=True)


for bath_datainputs, bath_datalabels in train_dl:
    break

model = SanaModel()
pred = model(bath_datainputs)[:, 0]

loss_fn =BCELoss()

print(1)
loss = loss_fn(pred, bath_datalabels.float())
print(loss.shape)
print(loss)
print()
print(2)
is_corret = (pred>=0.5)
print(is_corret)
print(3)
is_corret = is_corret.float()
print(is_corret)
print(4)
is_corret = ((pred>=0.5).float() == bath_datalabels).float()
print(is_corret)
print(5)
is_corret = is_corret.mean()
print(is_corret)
print(6)
is_corret=is_corret.item()
print(is_corret)
print()