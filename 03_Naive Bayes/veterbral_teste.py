from slave import *
import numpy as np
import matplotlib.pyplot as plt

# Carregar o conjunto de dados vertebral_column
X, y = load_vertebral_column_uci()

# Reduzir o número de dados para 50 por classe
X_reduced = []
y_reduced = []
for label in np.unique(y):
    X_class = X[y == label][:50]  # Selecionar os primeiros 50 dados de cada classe
    y_class = y[y == label][:50]
    X_reduced.extend(X_class)
    y_reduced.extend(y_class)

X_reduced = np.array(X_reduced)
y_reduced = np.array(y_reduced)

# Definir o número de realizações
num_realizacoes = 20

# Listas para armazenar acurácias
acuracias_knn = []
acuracias_dmc = []
acuracias_naive_bayes = []

# Loop sobre as realizações
for realizacao in range(num_realizacoes):
    # Dividir os dados em conjunto de treinamento e teste (Holdout)
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_reduced, test_size=0.3, random_state=realizacao)
    
    # Inicializar e treinar o classificador KNN
    knn = KNN()
    knn.fit(X_train[:, :2], y_train)
    y_pred_knn = knn.predict(X_test[:, :2])
    acuracia_knn = accuracy_score(y_test, y_pred_knn)
    acuracias_knn.append(acuracia_knn)

    # Imprimir a acurácia e o número da realização para KNN
    print("\nKNN - Realização", realizacao + 1)
    print("Acurácia:", acuracia_knn)
    print("Desvio Padrão:", np.std(acuracias_knn))
    print("Melhor Realização:", np.argmax(acuracias_knn) + 1)

    # Calcular e imprimir a matriz de confusão para KNN
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    print("Matriz de Confusão:")
    print(conf_matrix_knn)

    # Plotar a superfície de decisão do classificador KNN
    plt.figure(figsize=(8, 6))
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
    plt.title("Superfície de Decisão - KNN (Realização {})".format(realizacao + 1))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Inicializar e treinar o classificador DMC
    dmc = DMC()
    dmc.fit(X_train[:, :2], y_train)
    y_pred_dmc = dmc.predict(X_test[:, :2])
    acuracia_dmc = accuracy_score(y_test, y_pred_dmc)
    acuracias_dmc.append(acuracia_dmc)

    # Imprimir a acurácia e o número da realização para DMC
    print("\nDMC - Realização", realizacao + 1)
    print("Acurácia:", acuracia_dmc)
    print("Desvio Padrão:", np.std(acuracias_dmc))
    print("Melhor Realização:", np.argmax(acuracias_dmc) + 1)

    # Calcular e imprimir a matriz de confusão para DMC
    conf_matrix_dmc = confusion_matrix(y_test, y_pred_dmc)
    print("Matriz de Confusão:")
    print(conf_matrix_dmc)

    # Plotar a superfície de decisão do classificador DMC
    plt.figure(figsize=(8, 6))
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = dmc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
    plt.title("Superfície de Decisão - DMC (Realização {})".format(realizacao + 1))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Inicializar e treinar o classificador Naive Bayes
    naive_bayes = NaiveBayes()
    naive_bayes.fit(X_train[:, :2], y_train)
    y_pred_naive_bayes = naive_bayes.predict(X_test[:, :2])
    acuracia_naive_bayes = accuracy_score(y_test, y_pred_naive_bayes)
    acuracias_naive_bayes.append(acuracia_naive_bayes)

    # Imprimir a acurácia e o número da realização para Naive Bayes
    print("\nNaive Bayes - Realização", realizacao + 1)
    print("Acurácia:", acuracia_naive_bayes)
    print("Desvio Padrão:", np.std(acuracias_naive_bayes))
    print("Melhor Realização:", np.argmax(acuracias_naive_bayes) + 1)

    # Calcular e imprimir a matriz de confusão para Naive Bayes
    conf_matrix_naive_bayes = confusion_matrix(y_test, y_pred_naive_bayes)
    print("Matriz de Confusão:")
    print(conf_matrix_naive_bayes)

    # Plotar a superfície de decisão do classificador Naive Bayes
    plt.figure(figsize=(8, 6))
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = naive_bayes.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
    plt.title("Superfície de Decisão - Naive Bayes (Realização {})".format(realizacao + 1))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
