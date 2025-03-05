"""
dataFunction2.py

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import confusion_matrix, mean_absolute_error


def clasesDiferentes(df):
    print("\nClases diferentes en la columna dpi:")
    print(df['dpi'].unique())
    print("\nClases diferentes en la columna Sana:")
    print(df['Sana'].unique())
    print('Donde:\n')
    print('1 es Sana')
    print('0 es Fusarium')
    print('-1 es E_Hidrico')
    print("\nClases diferentes en la columna Tratamiento:")
    print(df['Tratamiento'].unique())
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
    ddf = df[['Tratamiento', 'dpi']].groupby('Tratamiento').count()
    nombres = ddf.index.values
    ddf = ddf.values.reshape(8,)
    colores = ["#EE6055","#60D394","#AAF683","#FFD97D","#FF9B85", "#0B4CA5", "#6083A4","#AAC683"]
    plt.pie(ddf, labels=nombres, autopct="%0.1f %%", colors=colores)
    plt.axis("equal")
    plt.show()


def bar_dpi(df):
    ddf = df[['dpi', 'Tratamiento']].groupby('dpi').count()
    index = ddf.index.values
    valores = ddf.values.reshape(-1,)
    n = valores.sum()
    valores = valores/n
    nombres = ['dpi ' + str(i) for i in index]
    
    plt.bar(nombres, valores, color='skyblue')
    plt.xlabel("DPI")
    plt.ylabel("Cantidad de Plantas")
    plt.title("Distribución de Plantas por DPI")
    plt.xticks(rotation=45)
    plt.show()

def resumen_Clases_dpi(df):
    n = df.shape[0]
    dpi = df['dpi'].unique()
    tratamiento = df['Tratamiento'].unique()
    print("\nPorcentaje de cada clase de tratamiento por cada dpi")
    for i in tratamiento:
        print(f'\nPorcentaje de la clase {i}:')
        for j in dpi:
            filtro = (df['Tratamiento'] == i) & (df['dpi'] == j)
            df_filtro = df[filtro]
            x = df_filtro.shape[0]
            p = (x/n)*100
            p = round(p, 1)
            print(f'{p}% en dpi {j}')


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



def explainedVarianceLongitudesOnda(df):
    X_ = df.iloc[:, 3:].values.T
    cov_M = np.cov(X_)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_M)
    eigen_vals = eigen_vals.real
    tot = sum(eigen_vals).real
    # la parte imaginaria de tot y eigen_vals es cero pero igual se las quito
    A = sorted(eigen_vals, reverse=True) 
    var_exp = [(i/tot) for i in A]
    var_exp = var_exp[:10]
    cum_var_exp = np.cumsum(var_exp)
    cum_var_exp = cum_var_exp[:10]
    x = [i+1 for i in range(len(var_exp))]
    print(cov_M.shape)
    print()
    print('Numero de componentes:', len(eigen_vals))
    print()
    print('Primeras 5 componentes de Explained variance ratio')
    print()
    s = 0
    for k in range(5):
        print(f'variance ratio {k+1}: {var_exp[k]}')
        s += var_exp[k]

    print(f'Sum explained variance ratio: {s}')


    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    ax.plot(x, var_exp, color='#DE0079', marker='o', markersize=5, label='Explained variance ratio')
    ax.step(x, cum_var_exp, where='post', color='#00C19B', marker='o', markersize=4, label='Cumulative explained variance')
    ax.set_xlabel('Principal component index', fontsize=12)
    ax.set_ylabel('Explained variance ratio', fontsize=12)

    plt.legend()
    plt.show()



def plot_datos_PCA2d(df):
    pca = PCA(n_components=2)
    X = df.iloc[:, 3:].values
    X = pca.fit(X).transform(X)
    df1 = df.iloc[:, :3]
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


def plot_heat_map(df):
    data = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap='plasma')

    cbar = ax.figure.colorbar(im, ax = ax)
    cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")

    plt.show()


def kfoldCV_clf(model, X, y, n_splits):
            kfold = KFold(n_splits=n_splits).split(X, y)
            scores = []
            for k, (train, test) in enumerate(kfold):
                model.fit(X[train], y[train])
                score = model.score(X[test], y[test])
                scores.append(score)
                print(f'fold {k+1}', f'accuracy {score:.3f}')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f'\nCV accuracy: {mean_score:.3f} +/- {std_score:.3f}')


def model_confmat(model, X, y):
    accuracy = model.score(X, y)
    print(f'\naccuracy: {accuracy:.3f}')
    y_predict = model.predict(X)
    confmat = confusion_matrix(y_pred=y_predict, y_true=y)
    print()
    print('Confusion Matrix:')
    print(confmat)
    clases = np.unique(y)

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
    plt.ylabel('True Label')
    plt.show()

def learningCurve_clf(model, X, y, n_splits):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, n_splits),
        cv=n_splits,
        n_jobs=-1,
        shuffle=False,
        random_state=33
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
        

def kfoldCV_reg(model, X, y, n_splits):
            kfold = KFold(n_splits=n_splits).split(X, y)
            scores = []
            for k, (train, test) in enumerate(kfold):
                model.fit(X[train], y[train])
                y_pred = model.predict(X[test])
                score = mean_absolute_error(y_true=y[test], y_pred=y_pred)
                scores.append(score)
                print(f'fold {k+1}', f'mean_absolute_error {score:.3f}')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f'\nCV mean_absolute_error: {mean_score:.3f} +/- {std_score:.3f}')


def plotRegModel(y_pred, y_df, Sanos):
    df = y_df.copy()
    df['dpi predict'] = y_pred

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    if Sanos == True:
        filtro = (df['Tratamiento'] == 'Control')
        dff = df[filtro]
        x = dff.index.values
        y_ = dff['dpi predict'].values
        y = dff['dpi'].values

        ax.plot(x, y, label='data Con', marker='x', linestyle='None', color="#028FA3")
        ax.plot(x, y_, label='predict Con', marker='x', linestyle='None', color="#FF8100")
        
    filtro = (df['Tratamiento'] != 'Control')
    dff = df[filtro]
    x = dff.index.values
    y_ = dff['dpi predict']
    y = dff['dpi']

    ax.plot(x, y, label='data', marker='.', linestyle='None', color="#028FA3")
    ax.plot(x, y_, label='predict', marker='.', linestyle='None', color="#FF8100")

    ax.set(xlabel='Sample index', ylabel='dpi: dias', title='dpi en test')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def learningCurve_reg(model, X, y, n_splits):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, n_splits),
        cv=n_splits,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        shuffle=False,
        random_state=33
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

