import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from slave import *
# Função para carregar a base de dados Iris
def load_iris():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    return X, y

# Função para dividir os dados em conjunto de treinamento e teste
def train_test_split(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Classe para o classificador KNN
class KNN:
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=3):
        y_pred = [self._predict(x, k) for x in X]
        return np.array(y_pred)

    def _predict(self, x, k):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Classe para o classificador DMC
class DMC:
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        nearest_index = np.argmin(distances)
        return self.y_train[nearest_index]

# Função para calcular a acurácia
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Função para calcular a matriz de confusão
def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]] += 1
    return matrix

# Carregar o conjunto de dados Iris
X, y = load_iris()

# Definir o número de realizações
num_realizacoes = 25

# Listas para armazenar acurácias
acuracias_knn = []
acuracias_dmc = []
acuracias_naive_bayes = []

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

    # Inicializar e treinar o classificador Naive Bayes
    def fit_naive_bayes(X_train, y_train):
        means = {}
        stds = {}
        class_labels = np.unique(y_train)
        for label in class_labels:
            means[label] = X_train[y_train == label].mean(axis=0)
            stds[label] = X_train[y_train == label].std(axis=0)
        return means, stds

    def predict_naive_bayes(X, means, stds):
        predictions = []
        class_labels = list(means.keys())
        for x in X:
            probs = []
            for label in class_labels:
                prob = np.sum(np.log((1 / (np.sqrt(2 * np.pi) * stds[label])) * np.exp(-((x - means[label]) ** 2) / (2 * (stds[label] ** 2)))))
                probs.append(prob)
            predictions.append(class_labels[np.argmax(probs)])
        return np.array(predictions)

    means, stds = fit_naive_bayes(X_train[:, :2], y_train)
    y_pred_naive_bayes = predict_naive_bayes(X_test[:, :2], means, stds)
    acuracia_naive_bayes = accuracy_score(y_test, y_pred_naive_bayes)
    acuracias_naive_bayes.append(acuracia_naive_bayes)

# Calcular a melhor acurácia e o desvio padrão correspondente para KNN
melhor_acuracia_knn = max(acuracias_knn)
melhor_desvio_padrao_knn = np.std(acuracias_knn)

# Calcular a melhor acurácia e o desvio padrão correspondente para DMC
melhor_acuracia_dmc = max(acuracias_dmc)
melhor_desvio_padrao_dmc = np.std(acuracias_dmc)

# Calcular a melhor acurácia e o desvio padrão correspondente para Naive Bayes
melhor_acuracia_naive_bayes = max(acuracias_naive_bayes)
melhor_desvio_padrao_naive_bayes = np.std(acuracias_naive_bayes)

# Encontrar a realização mais próxima da média para KNN
media_acuracia_knn = np.mean(acuracias_knn)
indice_realizacao_proxima_media_knn = np.argmin(np.abs(acuracias_knn - media_acuracia_knn))
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.3, random_state=indice_realizacao_proxima_media_knn)
knn_proximo_media = KNN()
knn_proximo_media.fit(X_train_knn[:, :2], y_train_knn)
y_pred_knn_proximo_media = knn_proximo_media.predict(X_test_knn[:, :2])
conf_matrix_knn = confusion_matrix(y_test_knn, y_pred_knn_proximo_media)

# Encontrar a realização mais próxima da média para DMC
media_acuracia_dmc = np.mean(acuracias_dmc)
indice_realizacao_proxima_media_dmc = np.argmin(np.abs(acuracias_dmc - media_acuracia_dmc))
X_train_dmc, X_test_dmc, y_train_dmc, y_test_dmc = train_test_split(X, y, test_size=0.3, random_state=indice_realizacao_proxima_media_dmc)
dmc_proximo_media = DMC()
dmc_proximo_media.fit(X_train_dmc[:, :2], y_train_dmc)
y_pred_dmc_proximo_media = dmc_proximo_media.predict(X_test_dmc[:, :2])
conf_matrix_dmc = confusion_matrix(y_test_dmc, y_pred_dmc_proximo_media)

# Encontrar a realização mais próxima da média para Naive Bayes
media_acuracia_naive_bayes = np.mean(acuracias_naive_bayes)
indice_realizacao_proxima_media_naive_bayes = np.argmin(np.abs(acuracias_naive_bayes - media_acuracia_naive_bayes))
means, stds = fit_naive_bayes(X_train[:, :2], y_train)
y_pred_naive_bayes_proximo_media = predict_naive_bayes(X_test[:, :2], means, stds)
conf_matrix_naive_bayes = confusion_matrix(y_test, y_pred_naive_bayes_proximo_media)

# Imprimir informações
print("Dataset: Iris")

# Imprimir informações para KNN
print("\nMelhor Realização para KNN (baseada na acurácia mais próxima da média):", indice_realizacao_proxima_media_knn)
print("Melhor Acurácia para KNN:", melhor_acuracia_knn)
print("Desvio Padrão para KNN:", melhor_desvio_padrao_knn)
print("Matriz de Confusão para KNN (Realização mais próxima da média):")
print(conf_matrix_knn)

# Imprimir informações para DMC
print("\nMelhor Realização para DMC (baseada na acurácia mais próxima da média):", indice_realizacao_proxima_media_dmc)
print("Melhor Acurácia para DMC:", melhor_acuracia_dmc)
print("Desvio Padrão para DMC:", melhor_desvio_padrao_dmc)
print("Matriz de Confusão para DMC (Realização mais próxima da média):")
print(conf_matrix_dmc)

# Imprimir informações para Naive Bayes
print("\nMelhor Realização para Naive Bayes (baseada na acurácia mais próxima da média):", indice_realizacao_proxima_media_naive_bayes)
print("Melhor Acurácia para Naive Bayes:", melhor_acuracia_naive_bayes)
print("Desvio Padrão para Naive Bayes:", melhor_desvio_padrao_naive_bayes)
print("Matriz de Confusão para Naive Bayes (Realização mais próxima da média):")
print(conf_matrix_naive_bayes)

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
plt.title("Dataset - IRIS\nSuperfície de Decisão - KNN")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

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
plt.title("Dataset - IRIS\nSuperfície de Decisão - DMC")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Plotar a superfície de decisão do classificador Naive Bayes
plt.figure(figsize=(8, 6))
h = .02  # step size in the mesh
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = predict_naive_bayes(np.c_[xx.ravel(), yy.ravel()], means, stds)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
plt.title("Dataset - IRIS\nSuperfície de Decisão - Naive Bayes")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
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

plt.title('Conjunto de Dados Iris - Distribuição das Classes e Pontos de Treinamento/Teste')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.legend()
plt.show()
