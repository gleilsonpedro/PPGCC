from slave import *
import numpy as np
import matplotlib.pyplot as plt


# Carregar o conjunto de dados vertebral_column
X, y = load_vertebral_column_uci()

# Definir o número de realizações
num_realizacoes = 25

# Listas para armazenar acurácias
acuracias_knn = []
acuracias_dmc = []
acuracias_bayes = []

# Loop sobre as realizações
for _ in range(num_realizacoes):
    # Dividir os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=_)
    
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

    # Inicializar e treinar o classificador Gaussian NB (Bayesiano)
    bayes = GaussianNB()
    bayes.fit(X_train[:, :2], y_train)
    y_pred_bayes = bayes.predict(X_test[:, :2])
    acuracia_bayes = accuracy_score(y_test, y_pred_bayes)
    acuracias_bayes.append(acuracia_bayes)

# Calcular a melhor acurácia e o desvio padrão correspondente para KNN
melhor_acuracia_knn = max(acuracias_knn)
melhor_desvio_padrao_knn = np.std(acuracias_knn)

# Calcular a melhor acurácia e o desvio padrão correspondente para DMC
melhor_acuracia_dmc = max(acuracias_dmc)
melhor_desvio_padrao_dmc = np.std(acuracias_dmc)

# Calcular a melhor acurácia e o desvio padrão correspondente para Gaussian NB (Bayesiano)
melhor_acuracia_bayes = max(acuracias_bayes)
melhor_desvio_padrao_bayes = np.std(acuracias_bayes)

# Imprimir o nome do dataset
print("Dataset: Vertebral Column")

# Imprimir a melhor acurácia e o desvio padrão para KNN
print("\nMelhor Acurácia e Desvio Padrão do KNN:")
print("Acurácia:", melhor_acuracia_knn)
print("Desvio Padrão:", melhor_desvio_padrao_knn)

# Encontrar a realização mais próxima da média para KNN
media_acuracia_knn = np.mean(acuracias_knn)
indice_realizacao_proxima_media_knn = np.argmin(np.abs(acuracias_knn - media_acuracia_knn))

# Imprimir a melhor realização para KNN
print("Melhor Realização KNN:", indice_realizacao_proxima_media_knn)

# Dividir novamente os dados para encontrar a realização mais próxima da média para KNN
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.3, random_state=indice_realizacao_proxima_media_knn)

# Inicializar e treinar o classificador KNN para a realização mais próxima da média
knn_proximo_media = KNN()
knn_proximo_media.fit(X_train_knn[:, :2], y_train_knn)
y_pred_knn_proximo_media = knn_proximo_media.predict(X_test_knn[:, :2])

# Calcular e imprimir a matriz de confusão para KNN
conf_matrix_knn = confusion_matrix(y_test_knn, y_pred_knn_proximo_media)
print("Matriz de Confusão para KNN (Realização mais próxima da média):")
print(conf_matrix_knn)

# Plotar a superfície de decisão do classificador KNN
plt.figure(figsize=(8, 6))
h = .02  # step size in the mesh
x_min, x_max = X_train_knn[:, 0].min() - 1, X_train_knn[:, 0].max() + 1
y_min, y_max = X_train_knn[:, 1].min() - 1, X_train_knn[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn_proximo_media.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_knn[:, 0], X_train_knn[:, 1], c=y_train_knn, s=20, edgecolor='k')
plt.title("Dataset - Vertebral Column\nSuperfície de Decisão - KNN")
plt.xlabel('Pelvic Incidence')
plt.ylabel('Pelvic Tilt')
plt.show()

# Imprimir a melhor acurácia e o desvio padrão para DMC
print("\nMelhor Acurácia e Desvio Padrão do DMC:")
print("Acurácia:", melhor_acuracia_dmc)
print("Desvio Padrão:", melhor_desvio_padrao_dmc)

# Encontrar a realização mais próxima da média para DMC
media_acuracia_dmc = np.mean(acuracias_dmc)
indice_realizacao_proxima_media_dmc = np.argmin(np.abs(acuracias_dmc - media_acuracia_dmc))

# Imprimir a melhor realização para DMC
print("Melhor Realização DMC:", indice_realizacao_proxima_media_dmc)

# Dividir novamente os dados para encontrar a realização mais próxima da média para DMC
X_train_dmc, X_test_dmc, y_train_dmc, y_test_dmc = train_test_split(X, y, test_size=0.3, random_state=indice_realizacao_proxima_media_dmc)

# Inicializar e treinar o classificador DMC para a realização mais próxima da média
dmc_proximo_media = DMC()
dmc_proximo_media.fit(X_train_dmc[:, :2], y_train_dmc)
y_pred_dmc_proximo_media = dmc_proximo_media.predict(X_test_dmc[:, :2])

# Calcular e imprimir a matriz de confusão para DMC
conf_matrix_dmc = confusion_matrix(y_test_dmc, y_pred_dmc_proximo_media)
print("\nMatriz de Confusão para DMC (Realização mais próxima da média):")
print(conf_matrix_dmc)

# Plotar a superfície de decisão do classificador DMC
plt.figure(figsize=(8, 6))
h = .02  # step size in the mesh
x_min, x_max = X_train_dmc[:, 0].min() - 1, X_train_dmc[:, 0].max() + 1
y_min, y_max = X_train_dmc[:, 1].min() - 1, X_train_dmc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = dmc_proximo_media.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_dmc[:, 0], X_train_dmc[:, 1], c=y_train_dmc, s=20, edgecolor='k')
plt.title("Dataset - Vertebral Column\nSuperfície de Decisão - DMC")
plt.xlabel('Pelvic Incidence')
plt.ylabel('Pelvic Tilt')
plt.show()

# Imprimir a melhor acurácia e o desvio padrão para Gaussian NB (Bayesiano)
print("\nMelhor Acurácia e Desvio Padrão do Gaussian NB (Bayesiano):")
print("Acurácia:", melhor_acuracia_bayes)
print("Desvio Padrão:", melhor_desvio_padrao_bayes)

# Encontrar a realização mais próxima da média para Gaussian NB (Bayesiano)
media_acuracia_bayes = np.mean(acuracias_bayes)
indice_realizacao_proxima_media_bayes = np.argmin(np.abs(acuracias_bayes - media_acuracia_bayes))

# Imprimir a melhor realização para Gaussian NB (Bayesiano)
print("Melhor Realização Gaussian NB (Bayesiano):", indice_realizacao_proxima_media_bayes)

# Dividir novamente os dados para encontrar a realização mais próxima da média para Gaussian NB (Bayesiano)
X_train_bayes, X_test_bayes, y_train_bayes, y_test_bayes = train_test_split(X, y, test_size=0.3, random_state=indice_realizacao_proxima_media_bayes)

# Inicializar e treinar o classificador Gaussian NB (Bayesiano) para a realização mais próxima da média
bayes_proximo_media = GaussianNB()
bayes_proximo_media.fit(X_train_bayes[:, :2], y_train_bayes)
y_pred_bayes_proximo_media = bayes_proximo_media.predict(X_test_bayes[:, :2])

# Calcular e imprimir a matriz de confusão para Gaussian NB (Bayesiano)
conf_matrix_bayes = confusion_matrix(y_test_bayes, y_pred_bayes_proximo_media)
print("\nMatriz de Confusão para Gaussian NB (Bayesiano) (Realização mais próxima da média):")
print(conf_matrix_bayes)

# Plotar a superfície de decisão do classificador Gaussian NB (Bayesiano)
plt.figure(figsize=(8, 6))
h = .02  # step size in the mesh
x_min, x_max = X_train_bayes[:, 0].min() - 1, X_train_bayes[:, 0].max() + 1
y_min, y_max = X_train_bayes[:, 1].min() - 1, X_train_bayes[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = bayes_proximo_media.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_bayes[:, 0], X_train_bayes[:, 1], c=y_train_bayes, s=20, edgecolor='k')
plt.title("Dataset - Vertebral Column\nSuperfície de Decisão - Gaussian NB (Bayesiano)")
plt.xlabel('Pelvic Incidence')
plt.ylabel('Pelvic Tilt')
plt.show()

# Plotar as gaussianas para cada classe e os conjuntos de dados de treinamento e teste
plt.figure(figsize=(12, 8))

# Plotar os pontos de treinamento e teste
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1, marker='o', label='Treinamento')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1, marker='x', label='Teste')

# Plotar as gaussianas para cada classe
for label in np.unique(y_train):
    X_class = X_train[y_train == label]
    mean = np.mean(X_class, axis=0)
    cov = np.cov(X_class.T)
    samples = np.random.multivariate_normal(mean, cov, 1000)
    plt.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.2, label=f'Gaussiana Classe {label}')

plt.title('Conjunto de Dados Vertebral Column - Distribuição das Classes e Pontos de Treinamento/Teste')
plt.xlabel('Pelvic Incidence')
plt.ylabel('Pelvic Tilt')
plt.legend()
plt.show()
