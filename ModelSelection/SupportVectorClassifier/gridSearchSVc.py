"""
gridSearchSVc.py

"""
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

def get_datos(random_state):
    file = 'datos.csv'
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion ver2"
    path.replace('\\', '/')
    path =  os.path.join(path, file)
    df = pd.read_csv(path, sep=';')
    X = df.iloc[:, 4:].values
    y = df['Sana'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test
print()
random_state=12
X_train, X_test, y_train, y_test = get_datos(random_state)
print('random_state:', random_state)
print()

clf = SVC()

C_range = [0.001, 0.01, 1, 5, 7, 10, 50]
param_grid = [
    {'C':C_range, 'kernel':['linear']}
]
print('param_grid:')
print(param_grid)
print()

gs = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=10,
    refit=True,
)

gs.fit(X_train, y_train)
print('best_score:', gs.best_score_)
print('best_params_', gs.best_params_)