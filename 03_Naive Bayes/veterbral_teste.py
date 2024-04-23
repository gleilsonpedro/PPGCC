from slave import *
import numpy as np
import matplotlib.pyplot as plt

# Carregar o conjunto de dados vertebral_column
X, y = load_vertebral_column_uci()

# Redução do número de pontos para os plots
X_reduced = []
y_reduced = []
for label in np.unique(y):
    indices = np.where(y == label)[0][:100]  # Reduzindo para 100 pontos por classe
    X_reduced.extend(X[indices])
    y_reduced.extend(y[indices])
X_reduced = np.array(X_reduced)
y_reduced = np.array(y_reduced)

# Definir o número de realizações
num_realizacoes = 20

# Listas para armazenar acurácias
acuracias_knn = []
acuracias_dmc = []
acuracias_bayes = []

# Loop sobre as realizações
for _ in range(num_realizacoes):
    # Dividir os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_reduced, test_size=0.3, random_state=_)
    
    # Inicializar e treinar o classificador KNN
    knn = KNN()
    knn.fit(X_train[:, :2], y_train)
    y_pred_knn = knn.predict(X_test[:, :2])
    acuracia_knn = accuracy_score(y_test, y_pred_knn)
    acuracias_knn.append(acuracia_knn)

    # Inicializar e treinar o classificador DMC
    dmc = DMC()
    dmc.fit(X_train[:, :2], y_train)
    y_pred_dmc = dmc.predict(X_test[:, :2])
    acuracia_dmc = accuracy_score(y_test, y_pred_dmc)
    acuracias_dmc.append(acuracia_dmc)

    # Inicializar e treinar o classificador Naive Bayes
    naive_bayes = NaiveBayes()
    naive_bayes.fit(X_train[:, :2], y_train)
    y_pred_bayes = naive_bayes.predict(X_test[:, :2])
    acuracia_bayes = accuracy_score(y_test, y_pred_bayes)
    acuracias_bayes.append(acuracia_bayes)

# Calcular a melhor acurácia e o desvio padrão correspondente para KNN
melhor_acuracia_knn = max(acuracias_knn)
melhor_desvio_padrao_knn = np.std(acuracias_knn)
indice_melhor_realizacao_knn = np.argmax(acuracias_knn)

# Calcular a melhor acurácia e o desvio padrão correspondente para DMC
melhor_acuracia_dmc = max(acuracias_dmc)
melhor_desvio_padrao_dmc = np.std(acuracias_dmc)
indice_melhor_realizacao_dmc = np.argmax(acuracias_dmc)

# Calcular a melhor acurácia e o desvio padrão correspondente para Naive Bayes
melhor_acuracia_bayes = max(acuracias_bayes)
melhor_desvio_padrao_bayes = np.std(acuracias_bayes)
indice_melhor_realizacao_bayes = np.argmax(acuracias_bayes)

# Imprimir o nome do dataset
print("Dataset: Vertebral Column")

# Imprimir a melhor realização e a acurácia/desvio padrão para KNN
print("\nMelhor Realização e Acurácia/Desvio Padrão para KNN:")
print("Melhor Realização:", indice_melhor_realizacao_knn)
print("Acurácia:", melhor_acuracia_knn)
print("Desvio Padrão:", melhor_desvio_padrao_knn)

# Imprimir a melhor realização e a acurácia/desvio padrão para DMC
print("\nMelhor Realização e Acurácia/Desvio Padrão para DMC:")
print("Melhor Realização:", indice_melhor_realizacao_dmc)
print("Acurácia:", melhor_acuracia_dmc)
print("Desvio Padrão:", melhor_desvio_padrao_dmc)

# Imprimir a melhor realização e a acurácia/desvio padrão para Naive Bayes
print("\nMelhor Realização e Acurácia/Desvio Padrão para Naive Bayes:")
print("Melhor Realização:", indice_melhor_realizacao_bayes)
print("Acurácia:", melhor_acuracia_bayes)
print("Desvio Padrão:", melhor_desvio_padrao_bayes)

# Plotar a superfície de decisão e os pontos de treinamento
classificadores = [knn, dmc, naive_bayes]
nomes_classificadores = ['KNN', 'DMC', 'Naive Bayes']
for i, clf in enumerate(classificadores):
    plt.figure(figsize=(8, 6))
    h = .1  # step size in the mesh
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_reduced, s=20, edgecolor='k')
    plt.title(f"Superfície de Decisão - {nomes_classificadores[i]}")
    plt.xlabel('Pelvic Incidence')
    plt.ylabel('Pelvic Tilt')
    plt.show()
