"""
plotLearningCurve.py

"""

import os
import pandas as pd
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np


def get_datos(file, path):
    path_ = path
    path_ = path_.replace('\\', '/')
    path_ = os.path.join(path_, file)
    df = pd.read_csv(path_, sep=';')
    y = df['dpi'].values
    X = df.iloc[:, 4:].values
    return X, y


def plot_learningCurve(file, path, random_state, estimator):
    X_train, y_train = get_datos(file, path)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        shuffle=True, 
        random_state=random_state
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training negMAE')
    plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation negMAE')
    plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training examples')
    plt.ylabel('neg_mean_absolute_error')
    plt.legend(loc='lower right')
    plt.show()

