from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
#import numpy as np
from CNNmodel import EspectroDataset, SanaModel, ModelTrainer
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
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test  = get_datos()

train_dataset = EspectroDataset(inputs=X_train, labels=y_train)
test_dataset = EspectroDataset(inputs=X_test, labels=y_test)

train_dl = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
test_dl = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

model = SanaModel()

loss_fn = BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

trainer = ModelTrainer(model, loss_fn, optimizer)

num_epochs = 250


trainer.train_model(train_dl, test_dl, num_epochs)
trainer.plot_learning_curve()