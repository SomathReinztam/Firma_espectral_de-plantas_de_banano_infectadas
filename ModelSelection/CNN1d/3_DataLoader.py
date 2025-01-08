"""
3_DataLoader.py

"""

from torch.utils.data import Dataset, DataLoader
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
#import numpy as np

class EspectroDataset(Dataset):
    def __init__(self, inputs, labels):
        """
        inputs es un array de numpy de tamaño (n_train, 1364)
        labels es un array de numpy de tamaño (n_train,)
        """
        self.inputs = torch.tensor(inputs.tolist(), dtype=torch.float32) # convertir a inputs en un tensor de tamaño torch.Size([n_train, 1364])
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long) # convertir a labels en un tensor de tamaño torch.Size([n_train])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.inputs[idx].unsqueeze(0) # x es un tensor de tamaño torch.Size([1, 1364])
        y = self.labels[idx] # y es un tensor de tamaño torch.Size([]) ejem: tensor(0.0655)
        return x, y


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
data_loarder = DataLoader(dataset=dataset, batch_size=8)

data_inputs, data_labels = next(iter(data_loarder))


print("Data inputs", data_inputs.shape, "\n", data_inputs)
print()
print("Data labels", data_labels.shape, "\n", data_labels)