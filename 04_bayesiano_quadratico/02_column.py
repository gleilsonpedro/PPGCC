import numpy as np
import pandas as pd
import os
import requests
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Função para calcular métricas
def calculate_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    conf_matrix = np.zeros((3, 3))
    for true, pred in zip(y_true, y_pred):
        conf_matrix[true, pred] += 1
    return accuracy, np.std(y_pred == y_true), conf_matrix

def load_vertebral_column_uci():
    # URL do arquivo ZIP
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
    # Caminho local para salvar o arquivo ZIP
    zip_path = "vertebral_column_data.zip"
    # Caminho local para o arquivo de dados extraído
    data_path = "column_3C.dat"

    # Baixar o arquivo ZIP se ainda não foi baixado
    if not os.path.exists(zip_path):
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)

    # Extrair o arquivo de dados do ZIP se ainda não foi extraído
    if not os.path.exists(data_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall()

    # Ler o arquivo de dados
    column_names = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis', 'class']
    vertebral_data = pd.read_csv(data_path, header=None, sep=' ', names=column_names)
    X = vertebral_data.iloc[:, :-1].values
    y = vertebral_data.iloc[:, -1].replace({'DH': 0, 'SL': 1, 'NO': 2}).values

    return X, y

# Load the Vertebral Column dataset
X, y = load_vertebral_column_uci()


# Holdout com 20 realizações
n_realizations = 20
accuracies = []
for _ in range(n_realizations):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    # Calcular class priors
    class_priors = np.bincount(y_train) / len(y_train)

    # Calcular class means e variances
    class_means = [np.mean(X_train[y_train == i], axis=0) for i in range(3)]
    class_variances = [np.var(X_train[y_train == i], axis=0) for i in range(3)]

    # Função de previsão
    def predict(X):
        predictions = []
        for x in X:
            class_scores = [np.log(class_priors[i]) - 0.5 * np.sum(np.log(2 * np.pi * class_variances[i]))
                            - 0.5 * np.sum(((x - class_means[i]) ** 2) / class_variances[i]) for i in range(3)]
            predictions.append(np.argmax(class_scores))
        return predictions

    # Fazer previsões no conjunto de teste
    y_pred = predict(X_test)

    # Calcular métricas
    accuracy, std_dev, conf_matrix = calculate_metrics(y_test, y_pred)
    accuracies.append((accuracy, std_dev, conf_matrix, X_train))

# Calcular média e desvio padrão das acurácias
mean_accuracies = np.mean([acc[0] for acc in accuracies])
std_accuracies = np.std([acc[0] for acc in accuracies])

# Encontrar a melhor realização com base na acurácia mais próxima da média
best_realization_index = np.argmin(np.abs([acc[0] - mean_accuracies for acc in accuracies]))
best_accuracy, best_std_dev, best_conf_matrix, best_X_train = accuracies[best_realization_index]

# Imprimir os resultados da melhor realização

print(f"Dataset : IRIS\n")
print(f"Melhor Realização: {best_realization_index}")
print(f"Acurácia: {best_accuracy:.4f}")
print(f"Desvio Padrão: {best_std_dev:.4f}")
print("Matriz de Confusão:")
print(best_conf_matrix)
print("Covariance Matrix (Complete):")
print(np.cov(best_X_train.T))
print("Covariance Matrix (Diagonal):")
print(np.diag(np.diag(np.cov(best_X_train.T))))
print("Covariance Matrix (Equal):")
print(np.mean(np.cov(best_X_train.T)) * np.eye(best_X_train.shape[1]))

#PLOTS

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Função para plotar a superfície de decisão
def plot_decision_surface(X, y, means, variances, priors, ax):
    h = 0.02  # Passo do grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(np.c_[xx.ravel(), yy.ravel()], means, variances, priors)
    Z = np.array(Z).reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)

# Função de previsão
def predict(X, means, variances, priors):
    predictions = []
    for x in X:
        class_scores = [np.log(priors[i]) - 0.5 * np.sum(np.log(2 * np.pi * variances[i]))
                        - 0.5 * np.sum(((x - means[i][:2]) ** 2) / variances[i][:2]) for i in range(3)]
        predictions.append(np.argmax(class_scores))
    return predictions

# Dados para plotagem
X_plot = best_X_train[:, :2]  # Usando apenas as duas primeiras características
y_plot = y_train

# Plotagem
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_surface(X_plot, y_plot, class_means, class_variances, class_priors, ax)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Dataset - IRIS\nSuperfície de Decisão - Melhor Realização')
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define custom colors for training and test points
train_colors = ['gold', 'limegreen', 'royalblue']
test_color = 'black'

def plot_dataset(X_train, y_train, X_test, y_test):
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.scatter(X_train[y_train == i][:, 0], X_train[y_train == i][:, 1], c=train_colors[i], marker='o', label=f'{iris.target_names[i]} (Treinamento)', edgecolors='k', s=80)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='Teste', edgecolors='k', s=80)
    plt.xlabel('Comprimento da Sépala')
    plt.ylabel('Largura da Sépala')
    plt.title('Dataset - IRIS\nConjuntos de Dados de Treinamento e Teste')

    # Adicionando uma legenda personalizada com nomes de espécies
    custom_lines = [Line2D([0], [0], marker='o', color='w', label=f'{iris.target_names[i]} (Treinamento)', markerfacecolor=train_colors[i], markersize=10) for i in range(3)]
    custom_lines.append(Line2D([0], [0], marker='x', color='black', label='Teste', markersize=10))
    plt.legend(handles=custom_lines, loc='upper right')

    plt.show()

# Usar a função para plotar os conjuntos de dados de treinamento e teste
plot_dataset(X_train, y_train, X_test, y_test)
