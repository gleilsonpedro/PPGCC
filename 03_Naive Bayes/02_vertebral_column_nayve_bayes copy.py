import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import pandas as pd
import os
import requests
import zipfile

# Função para carregar os dados do conjunto de dados vertebral column da UCI
def load_vertebral_column_uci():
    # Criar o diretório 'dados' se ele não existir
    if not os.path.exists('dados'):
        os.makedirs('dados')

    # URL do arquivo ZIP
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
    # Caminho local para salvar o arquivo ZIP
    zip_path = "dados/vertebral_column_data.zip"
    # Caminho local para o arquivo de dados extraído
    data_path = "dados/column_3C.dat"

    # Baixar o arquivo ZIP se ainda não foi baixado
    if not os.path.exists(zip_path):
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)

    # Extrair o arquivo de dados do ZIP se ainda não foi extraído
    if not os.path.exists(data_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("dados")

    # Ler o arquivo de dados
    column_names = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis', 'class']
    vertebral_data = pd.read_csv(data_path, header=None, sep=' ', names=column_names)
    X = vertebral_data.iloc[:, :-1].values
    y = vertebral_data.iloc[:, -1].replace({'DH': 0, 'SL': 1, 'NO': 2}).values

    return X, y

# Função para treinar o classificador Naive Bayes
def train_naive_bayes(X, y):
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    n_features = X.shape[1]
    prior_probs = np.zeros(n_classes)
    means = np.zeros((n_classes, n_features))
    stds = np.zeros((n_classes, n_features))

    # Calcular as probabilidades a priori para cada classe
    for i, label in enumerate(class_labels):
        prior_probs[i] = np.mean(y == label)

        # Calcular as médias e os desvios padrão para cada atributo em cada classe
        means[i] = np.mean(X[y == label], axis=0)
        stds[i] = np.std(X[y == label], axis=0)

    return prior_probs, means, stds

# Função para fazer previsões com o classificador Naive Bayes
def predict_naive_bayes(X, prior_probs, means, stds):
    n_samples, n_features = X.shape
    n_classes = len(prior_probs)
    likelihood_probs = np.zeros((n_samples, n_classes))

    for i in range(n_samples):
        for j in range(n_classes):
            likelihood_probs[i, j] = np.prod(1 / (np.sqrt(2 * np.pi) * stds[j]) * np.exp(-(X[i] - means[j]) ** 2 / (2 * stds[j] ** 2)))

    posterior_probs = likelihood_probs * prior_probs
    y_pred = np.argmax(posterior_probs, axis=1)
    
    return y_pred

# Função para dividir os dados em treino e teste e executar o holdout com 20 realizações
def holdout(X, y, test_size=0.3, random_state=42, num_runs=20):
    accuracies_knn = []
    accuracies_dmc = []
    accuracies_nb = []

    for run in range(1, num_runs + 1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Treinamento e teste do k-NN
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        accuracies_knn.append(accuracy_knn)

        # Treinamento e teste do DMC (Discriminante de Mínima Distância)
        dmc = NearestCentroid()
        dmc.fit(X_train, y_train)
        y_pred_dmc = dmc.predict(X_test)
        accuracy_dmc = accuracy_score(y_test, y_pred_dmc)
        accuracies_dmc.append(accuracy_dmc)

        # Treinamento e teste do Naive Bayes
        prior_probs, means, stds = train_naive_bayes(X_train, y_train)
        y_pred_nb = predict_naive_bayes(X_test, prior_probs, means, stds)
        accuracy_nb = accuracy_score(y_test, y_pred_nb)
        accuracies_nb.append(accuracy_nb)

    # Calculando a média e desvio padrão das acurácias
    avg_accuracy_knn = np.mean(accuracies_knn)
    std_accuracy_knn = np.std(accuracies_knn)

    avg_accuracy_dmc = np.mean(accuracies_dmc)
    std_accuracy_dmc = np.std(accuracies_dmc)

    avg_accuracy_nb = np.mean(accuracies_nb)
    std_accuracy_nb = np.std(accuracies_nb)

    # Encontrando a melhor realização com base na acurácia média
    best_knn = (np.max(accuracies_knn), avg_accuracy_knn, std_accuracy_knn)
    best_dmc = (np.max(accuracies_dmc), avg_accuracy_dmc, std_accuracy_dmc)
    best_nb = (np.max(accuracies_nb), avg_accuracy_nb, std_accuracy_nb)

    return best_knn, best_dmc, best_nb

# Carregar os dados
X, y = load_vertebral_column_uci()

# Executar o holdout com 20 realizações
best_knn, best_dmc, best_nb = holdout(X, y)

# Exibir os resultados
print("\nMelhor acurácia para k-NN:")
print("Acurácia máxima:", best_knn[0])
print("Média das acurácias:", best_knn[1])
print("Desvio padrão das acurácias:", best_knn[2])

print("\nMelhor acurácia para DMC:")
print("Acurácia máxima:", best_dmc[0])
print("Média das acurácias:", best_dmc[1])
print("Desvio padrão das acurácias:", best_dmc[2])

print("\nMelhor acurácia para Naive Bayes:")
print("Acurácia máxima:", best_nb[0])
print("Média das acurácias:", best_nb[1])
print("Desvio padrão das acurácias:", best_nb[2])

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_surface(X, y, classifier, title):
    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)

    plt.show()

