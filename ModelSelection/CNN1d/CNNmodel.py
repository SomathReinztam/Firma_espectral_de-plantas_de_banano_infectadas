
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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

class SanaModel2(nn.Module):
    def __init__(self):
        super(SanaModel2, self).__init__()
        """
        Bloque convulucional

        """
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.3)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=12, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.4)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=16, kernel_size=8, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.4)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=20, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.3)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=24, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=57)
        )

        self.flatten = nn.Flatten()

        """
        Bloque fully conected

        """

        self.fc = nn.Sequential(
            nn.Linear(in_features=24, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class SanaModel(nn.Module):
    def __init__(self):
        super(SanaModel, self).__init__()
        """
        Bloque convulucional

        """
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.3)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=12, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.4)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=16, kernel_size=8, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.4)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=20, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.3)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=24, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=57)
        )

        self.flatten = nn.Flatten()

        """
        Bloque fully conected

        """

        self.fc = nn.Sequential(
            nn.Linear(in_features=24, out_features=16),
            nn.Softplus(),
            nn.Linear(in_features=16, out_features=8),
            nn.Tanh(),
            nn.Linear(in_features=8, out_features=10),
            nn.Softplus(),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model  # Cnn
        self.loss_fn = loss_fn # funcion de perdida
        self.optimizer = optimizer # optimizador 
        self.loss_hist_train = []
        self.accuracy_hist_train = []
        self.loss_hist_test = []
        self.accuracy_hist_test = []

    def train_model(self, train_dl, test_dl, num_epochs):
        for epoch in range(num_epochs):
            self.model.train() # poner el modelo en modo entrenamiento
            epoch_losses_train = []
            epoch_accuracies_train = []

            """
            bath_data_inputs es de tamaño torch.Size([8, 1, 1364])
            bath_data_labels es de tamaño torch.Size([8])

            """
            for bath_data_inputs, bath_data_labels in train_dl:
                pred = self.model(bath_data_inputs)[:, 0] # model(bath_data_inputs)->torch.Size([8, 1])  pred=model(bath_data_inputs)[:, 0]->torch.Size([8])
                # loss es la loss de un bath
                loss = self.loss_fn(pred, bath_data_labels.float()) # loss->torch.Size([]) ejem: tensor(0.5745)
                loss.backward() # backpropagation: calcula la gradiente con respecto a los pesos
                self.optimizer.step() # updates los pesos
                self.optimizer.zero_grad() # limpia la gradiente

                epoch_losses_train.append(loss.item()) 
                """
                pred>=0.5 significa que entrada por entrada 
                se clasificara como  1 en otro caso
                se clasificara como 0
                """
                is_correct = ((pred>=0.5).float() == bath_data_labels).float() # is_correct->torch.Size([8]) es un tensor que sus entradas son 0 y 1
                epoch_accuracies_train.append(is_correct.mean().item()) # is_correct.mean().item() se puese interpretar como el % de acirtos de el bath en el que estamos
                bath_data_inputs, bath_data_labels, pred = None, None, None # ahorrar memoria

                """
                Ahora en epoch_losses_train tiene la loss de todos los bath 
                y epoch_accuracies_train tien el %  de acirtos de todos los bath
                """
            self.loss_hist_train.append(np.mean(epoch_losses_train))
            self.accuracy_hist_train.append(np.mean(epoch_accuracies_train))

            test_loss, test_accuracy = self.validate(train_dl)
            self.loss_hist_test.append(test_loss)
            self.accuracy_hist_test.append(test_accuracy)
            print(f'Epoch {epoch+1} / {num_epochs}: Training Loss: {np.mean(epoch_losses_train):.4f}, Training Accuracy: {np.mean(epoch_accuracies_train):.4f}, Test Loss: {test_loss:.4f}, test Accuracy: {test_accuracy:.4f}')


    def validate(self, test_dl):
        self.model.eval()
        epoch_losses_test = []
        epoch_accuracies_test = []

        # desactivar los gradientes (grafo computacional)
        with torch.no_grad():
            # bath_data_inputs, bath_data_labels, y lo demas tienen el mismo tamaño de arriba
            for bath_data_inputs, bath_data_labels in test_dl: 
                pred = self.model(bath_data_inputs)[:, 0]
                loss = self.loss_fn(pred, bath_data_labels.float())
                epoch_losses_test.append(loss.item())
                is_correct = ((pred>=0.5).float() == bath_data_labels).float()
                epoch_accuracies_test.append(is_correct.mean().item())
                return np.mean(epoch_losses_test), np.mean(epoch_accuracies_test)
    

    def plot_learning_curve(self):
        plt.style.use('seaborn-whitegrid')
        fig, (ax0, ax1) = plt.subplots(ncols=2,  figsize=(12, 10))
        ax0.plot(self.loss_hist_train, linestyle='solid', label='Training')
        ax0.plot(self.loss_hist_test, linestyle='solid', label='Test')
        ax0.set_title("Loss history")
        ax0.legend()

        ax1.plot(self.accuracy_hist_train, linestyle='solid', label='Training')
        ax1.plot(self.accuracy_hist_test, linestyle='solid', label='Test')
        ax1.set_title("accuracy history")
        ax1.set_title("Accuracy history")
        ax1.legend()

        plt.show()




