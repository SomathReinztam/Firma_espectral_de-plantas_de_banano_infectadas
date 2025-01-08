"""
dataFunction.py

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import learning_curve, train_test_split

def clasesDiferentes(df):
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    print("\nClases diferentes en la columna dpi:")
    print(df['dpi'].unique())
    print("\nClases diferentes en la columna Sana:")
    print(df['Sana'].unique())
    print("\nClases diferentes en la columna Tratamiento:")
    print(df['Tratamiento'].unique())
    print('Donde:', d)
    print("\nClases diferentes en la columna Planta:")
    print(df['Planta'].unique())
    print()


def pie_Sanas(df):
    ddf = df[['Sana', 'dpi']].groupby('Sana').count()
    ddf = ddf.values.reshape(2,)
    nombres = ['No sana', 'Sana']
    colores = ["#EE6055","#60D394"]
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


def plot_datos_PCA_2d(df_):
    pca = PCA(n_components=2)
    X_ = df_.iloc[:, 4:].values
    X_ = pca.fit(X_).transform(X_)
    df1 = df_.iloc[:, :4]
    df2 = pd.DataFrame(X_, columns=['eje_x', 'eje_y']) 
    df = pd.concat([df1, df2], axis=1)

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    estilos = ['x' ,'d', '^', 's', '+']
    colores = ['#2F900B', '#2A9B00', '#3BDB00', '#4FD51E', '#2A9B00']
    Tratamiento_class = [1, 2, 3, 4, 5]
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    n = len(Tratamiento_class)

    filtro = (df['Sana'] == 0)
    df_filtro = df[filtro]
    x = df_filtro['eje_x']
    y = df_filtro['eje_y']
    ax.scatter(x, y, color='#FF7800', marker='o', label="No Sana")

    filtro = (df['Tratamiento'] == 0) & (df['dpi'] == 0)
    df_filtro = df[filtro]
    x = df_filtro['eje_x']
    y = df_filtro['eje_y']
    ax.scatter(x, y, color='#9FF700', marker='o', label="Con")


    for i in range(n):
        Tc = Tratamiento_class[i]
        filtro = (df['Tratamiento'] == Tc) & (df['dpi'] == 0)
        df_filtro = df[filtro]
        x = df_filtro['eje_x']
        y = df_filtro['eje_y']
        c = colores[i]
        s = estilos[i]
        ax.scatter(x, y, color=c, marker=s, label="{0}".format(d[Tc]))

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def plot_heat_map(df):
    data = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='plasma')

    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")

    plt.show()


def ver_plantas_filtro(df, filtro):
    lgtd_onda = [float(i) for i in df.columns.values[4:]]
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    df_filtro = df[filtro]
    m = df_filtro.shape[0]

    for i in range(m):
        serie = df_filtro.iloc[i, :]
        dpi = int(serie['dpi'])
        T = int(serie['Tratamiento'])
        T = d[T]
        Planta = int(serie['Planta'])
        y = serie[4:].values
        
        ax.plot(lgtd_onda, y, linewidth=1.5, label="dpi:{0} {1} P:{2}".format(dpi, T, Planta))
    
    plt.legend()
    plt.show()

def ver_planta(df, Tratamiento, Planta):
    lgtd_onda = [float(i) for i in df.columns.values[4:]]
    d = dict([(0, 'Con'), (1, 'Fo5'), (2, 'IsB'), (3, 'Var'), (4, 'ViE'), (5, 'HyS')])
    #     verde_1    verde_2     naranja    rojo
    c = ['#00D700', '#BBE200', '#FF8E00', '#EA1B00']

    filtro = (df['Tratamiento'] == Tratamiento) & (df['Planta'] == Planta)
    df_filtro = df[filtro]
    n = df_filtro.shape[0]

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    for i in range(n):
        serie = df_filtro.iloc[i, :]
        dpi = int(serie['dpi'])
        T = d[Tratamiento]
        y = serie[4:].values

        ax.plot(
            lgtd_onda, 
            y, 
            color=c[i], 
            linewidth=1.5,
            label="dpi:{0} {1} P:{2}".format(dpi, T, Planta)
        )
        
    plt.legend()
    plt.show()

def plot_describe(df_describe):
    lgtd_onda = [float(i) for i in df_describe.columns.values[5:]]
    L = [(1, 2), (3, 7), (4, 5, 6)]
    d = [(1, 'mean'), (2, 'std'), (3, 'min'), (7, 'max'), (4, '25%'), (5, '50%'), (6,'75%')]
    d = dict(d)

    plt.style.use('seaborn-whitegrid')
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,  figsize=(10, 6))
    axs = (ax0, ax1, ax2)
    a = 0
    for ax in axs:
        for i in L[a]:
            y = df_describe.iloc[i, 5:]
            ax.plot(lgtd_onda, y, label=d[i])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        a += 1
        
    
    plt.show()


def ver_mean_std(df, filtro):
    df_filtro = df[filtro]
    n = df_filtro.shape[0]
    lgtd_onda = [float(i) for i in df_filtro.columns.values[4:]]

    plt.style.use('seaborn-whitegrid')
    fig, (ax0, ax1) = plt.subplots(nrows=2,  figsize=(12, 6))

    for i in range(n):
        y = df_filtro.iloc[i, 4:]
        ax0.plot(lgtd_onda, y, color='#99B098')
    ax0.plot(lgtd_onda, y, color='#99B098', label='sample')
    y = df_filtro.iloc[:, 4:].mean(axis=0).values
    ax0.plot(lgtd_onda, y, color='#CF0079', linewidth=2.5, label='mean')
    ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))

    y = df_filtro.iloc[:, 4:].std(axis=0).values
    ax1.plot(lgtd_onda, y, color='#32AFA2', linewidth=2.5, label='std')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()

def ver_Means_Stds_free(df, filtros):
    lgtd_onda = [float(i) for i in df.columns.values[4:]]

    plt.style.use('seaborn-whitegrid')
    fig, (ax0, ax1) = plt.subplots(nrows=2,  figsize=(12, 6))

    x = 0
    for filtro in filtros:
        df_filtro = df[filtro]
        y1 = df_filtro.iloc[:, 4:].mean(axis=0).values
        y2 = df_filtro.iloc[:, 4:].std(axis=0).values

        ax0.plot(lgtd_onda, y1, linewidth=1.5, label="mean filtro {0}".format(x))
        ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.plot(lgtd_onda, y2, linewidth=1.5, label="std filtro {0}".format(x))
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        x += 1

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
        matriz = [["TP", "FN"],["FP", "TN"]]
        print()
        for fila in matriz:
            print(" ".join(fila))
        
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j],
                        va='center', ha='center')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('Predicted label')
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

