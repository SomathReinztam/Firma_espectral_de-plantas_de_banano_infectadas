"""
dataFunction2.py

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import confusion_matrix, mean_absolute_error

def reNombrarSana(df):
    cond1 = (df['dpi'] == 0) | (df['Tratamiento'] == 0)
    cond2 = (df['Tratamiento'] == 5)

    df['Sana'] = 0
    df.loc[cond2, df.columns[1]] = -1
    df.loc[cond1, df.columns[1]] = 1
    


def clasesDiferentes(df):
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    print("\nClases diferentes en la columna dpi:")
    print(df['dpi'].unique())
    print("\nClases diferentes en la columna Sana:")
    print(df['Sana'].unique())
    print('Donde:\n')
    print('1 es Sana')
    print('0 es Fusarium')
    print('-1 es HyS')
    print("\nClases diferentes en la columna Tratamiento:")
    print(df['Tratamiento'].unique())
    print('Donde:\n')
    for i in range(6):
        print(f'{i} es {d[i]}')
    print("\nClases diferentes en la columna Planta:")
    print(df['Planta'].unique())
    print()


def pie_Sanas(df):
    ddf = df[['Sana', 'dpi']].groupby('Sana').count()
    ddf = ddf.values.reshape(3,)
    nombres = ['HyS', 'Fusarium', 'Sana']
    colores = ['#830BD9', "#EE6055","#60D394"]
    plt.pie(ddf, labels=nombres, autopct="%0.1f %%", colors=colores)
    plt.axis("equal")
    plt.show()

def pie_Clases(df):
    d = {0: 'Con', 1: 'Fo5', 2: 'IsB', 3: 'Var', 4: 'ViE', 5: 'HyS'}
    ddf = df[['Tratamiento', 'dpi']].groupby('Tratamiento').count()
    ddf = ddf.values.reshape(6,)
    nombres = [d[i] for i in range(6)]
    colores = ["#EE6055","#60D394","#AAF683","#FFD97D","#FF9B85", "#0B4CA5"]
    plt.pie(ddf, labels=nombres, autopct="%0.1f %%", colors=colores)
    plt.axis("equal")
    plt.show()

def pie_dpi(df):
    ddf = df[['dpi', 'Planta']].groupby('dpi').count()
    index = ddf.index.values
    ddf = ddf.values.reshape(4,)
    nombres = ['dpi '+str(i) for i in index]
    colores = ["#EE6055","#60D394","#FFD97D", "#0B4CA5"]
    plt.pie(ddf, labels=nombres, autopct="%0.1f %%", colors=colores)
    plt.axis("equal")
    plt.show()

def resumen_Clases_dpi(df):
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    n = df.shape[0]
    print("\nPorcentaje de cada clase de tratamiento por cada dpi")
    l = [0, 7, 14, 21]
    for i in range(6):
        t = d[i]
        print('\nPorcentaje de la clase {0}:'.format(t))
        for j in l:
            filtro = (df['Tratamiento'] == i) & (df['dpi'] == j)
            df_filtro = df[filtro]
            x = df_filtro.shape[0]
            p = (x/n)*100
            p = round(p, 1)
            print('{0}% en dpi: {1}'.format(p, j))


def plot_datos_PCA2d(df):
    pca = PCA(n_components=2)
    X = df.iloc[:, 4:].values
    X = pca.fit(X).transform(X)
    df1 = df.iloc[:, :4]
    df2 = pd.DataFrame(X, columns=['eje_x', 'eje_y'])
    ddf = pd.concat([df1, df2], axis=1)

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    Sana = [0, 1, -1]
    Colores = ['#FF7400', '#3EE600', '#0B29C9']
    Label = ['Fusarium', 'Sana', 'HyS']
    for i in range(len(Sana)):
        filtro = (ddf['Sana'] == Sana[i])
        df_filtro = ddf[filtro]
        x = df_filtro['eje_x'].values
        y = df_filtro['eje_y'].values
        ax.scatter(x, y, color=Colores[i], marker='o', label=Label[i])
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()



class modelPerformance:
    def __init__(self, model):
        self.model = model
    
    def kfoldCV_clf(self, X_train, X_test, y_train, y_test, n_splits):
        kfold = KFold(n_splits).split(X_train, y_train)
        scores = []
        for k, (train, test) in enumerate(kfold):
            self.model.fit(X_train[train], y_train[train])
            score = self.model.score(X_train[test], y_train[test])
            score = round(score, 3)
            scores.append(score)
            print(
                'folt {}'.format(k+1),
                'acc {}'.format(score)
            )

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f'\nCV accuracy: {mean_score:.3f} +/- {std_score:.3f}')

        self.model.fit(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        print(f'\ntest accuracy: {test_acc:.3f}')

        print(f'\n confusion matrix:')
        y_predict = self.model.predict(X_test)
        confmat = confusion_matrix(y_pred=y_predict, y_true=y_test)
        print(confmat)
        
        clases = np.unique(y_predict)
        
        plt.style.use('classic')
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes()

        ax.matshow(confmat, cmap='viridis', alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
        n_labels = [i for i in range(confmat.shape[0])]
        ax.set_xticks(n_labels)
        ax.set_yticks(n_labels)
        ax.set_xticklabels(clases)
        ax.set_yticklabels(clases)

        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('Predict label')
        plt.ylabel('True label')
        plt.show()        

    def learningCurve_clf(self, X_train, y_train):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.model,
            X=X_train,
            y=y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=10,
            n_jobs=1,
            shuffle=False,
            random_state=12
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes()

        ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training acc')
        ax.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
        ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation acc')
        ax.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')

        ax.set_xlabel('Training Set Size', fontsize=12)  # Etiqueta para el eje X
        ax.set_ylabel('Accuracy', fontsize=12)  # Etiqueta para el eje Y
        #ax.legend(loc='best')
        ax.set_title('Learning Curve', fontsize=14)
        plt.legend()
        plt.show()

    def kfoldCV_reg(self, X_train, X_test, y_train, y_test, n_splits):
        kfold = KFold(n_splits).split(X_train, y_train)
        scores = []
        for k, (train, test) in enumerate(kfold):
            self.model.fit(X_train[train], y_train[train])
            y_predict = self.model.predict(X_train[test])
            score = mean_absolute_error(y_pred=y_predict, y_true=y_train[test])
            score = round(score, 3)
            scores.append(score)
            print(
                'folt {}'.format(k+1),
                'mean_absolute_error {}'.format(score)
            )

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f'\nCV mean_absolute_error: {mean_score:.3f} +/- {std_score:.3f}')

        self.model.fit(X_train, y_train)
        y_predict = self.model.predict(X_test)
        test_mae = mean_absolute_error(y_pred=y_predict, y_true=y_test)
        print(f'test mean_absolute_error: {test_mae:.3f}')


        1

    def learningCurve_reg(self, X_train, y_train):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.model,
            X=X_train,
            y=y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=10,
            scoring='neg_mean_absolute_error',
            n_jobs=1,
            shuffle=False,
            random_state=12
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes()

        ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training MAE')
        ax.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
        ax.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation MAE')
        ax.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')

        ax.set_xlabel('Training Set Size', fontsize=12)  # Etiqueta para el eje X
        ax.set_ylabel('neg_mean_absolute_error', fontsize=12)  # Etiqueta para el eje Y
        #ax.legend(loc='best')
        ax.set_title('Learning Curve', fontsize=14)
        plt.legend()
        plt.show()


