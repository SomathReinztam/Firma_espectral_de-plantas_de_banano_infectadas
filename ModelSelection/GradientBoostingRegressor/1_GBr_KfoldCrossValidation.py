"""
1_GBr_KfoldCrossValidation.py

"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


def get_datos(random_state):
    file = 'datos.csv'
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion ver2"
    path = path.replace('\\', '/')
    path = os.path.join(path, file)

    df = pd.read_csv(path, sep=';')
    y = df['Sana'].values
    X = df.iloc[:, 4:].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

print()
random_state=12
print('random_state: {}'.format(random_state))
print()
X_train, X_test, y_train, y_test = get_datos(random_state)

X_train, X_test, y_train, y_test = get_datos(random_state)
print('X_train: {0}, X_test: {1}, y_train: {2}, y_test: {3}'.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
print()

hyperparameter = {'loss':'absolute_error', 'n_estimators': 900, 'max_depth': 11, 'criterion': 'squared_error', 'max_features': 'sqrt'}
print('hyper parameters: {}'.format(hyperparameter))
print()

clf = GradientBoostingRegressor(**hyperparameter)

kfold = KFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    clf.fit(X_train[train], y_train[train])
    y_predict = clf.predict(X_train[test])
    score = mean_absolute_error(y_pred=y_predict, y_true=y_train[test])
    scores.append(score)
    print(
        'fold {}'.format(k+1),
        'mean_absolute_error {}'.format(score)
    )

mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f'\nCV mean_absolute_error: {mean_acc:.3f} +/- {std_acc:.3f}')

clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
test_accuracy = mean_absolute_error(y_pred=y_predict, y_true=y_test)
print(f'\ntest mean_absolute_error: {test_accuracy}')
print()




