import numpy as np
import pandas as pd
from collections import Counter
import zipfile
import requests
import os

###########################################################
#                    FUNÇÕES IRIS                         #

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

#######################################################################################
#                               FUNÇÕES VERTEBRAL COLUMN


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

class GaussianNB:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_priors = np.zeros(len(self.classes))
        self.class_means = np.zeros((len(self.classes), X_train.shape[1]))
        self.class_vars = np.zeros((len(self.classes), X_train.shape[1]))

        for i, c in enumerate(self.classes):
            X_c = X_train[y_train == c]
            self.class_priors[i] = X_c.shape[0] / X_train.shape[0]
            self.class_means[i] = np.mean(X_c, axis=0)
            self.class_vars[i] = np.var(X_c, axis=0) + 1e-9  # Adding a small value to avoid division by zero

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        n_features = X_test.shape[1]
        n_classes = len(self.classes)

        probs = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            prior = np.log(self.class_priors[i])
            class_mean = self.class_means[i]
            class_var = self.class_vars[i]
            likelihood = -0.5 * np.sum(np.log(2. * np.pi * class_var)) - \
                         0.5 * np.sum(((X_test - class_mean) ** 2) / (class_var), axis=1)
            class_probs = prior + likelihood
            probs[:, i] = class_probs

        return self.classes[np.argmax(probs, axis=1)]


# Implementação do classificador Naive Bayes
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        self.X_train = X  # Armazenar os dados de treinamento
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                "mean": X_c.mean(axis=0),
                "std": X_c.std(axis=0) + 1e-10,  # Adicionando um pequeno valor para evitar divisão por zero
            }

    def _pdf(self, X, mean, std):
        return np.exp(-0.5 * ((X - mean) / std) ** 2) / (np.sqrt(2 * np.pi) * std)

    def _predict_class(self, x):
        posteriors = []
        for c in self.classes:
            prior = len(self.parameters[c]["mean"]) / len(self.X_train)  # Corrigindo para usar self.X_train
            likelihood = np.sum(np.log(self._pdf(x, self.parameters[c]["mean"], self.parameters[c]["std"])))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._predict_class(x) for x in X]
        return np.array(y_pred)


#######################################
#              ARTIFICIAL             #
#######################################


# Definição da função para gerar o conjunto de dados artificial
def generate_artificial_dataset():
    np.random.seed(42)

    # Parâmetros para a Classe 1
    mean1 = [1, 1]
    cov1 = [[0.1, 0], [0, 0.1]]
    class1 = np.random.multivariate_normal(mean1, cov1, 10)

    # Parâmetros para a Classe 0
    mean2 = [0, 0]
    cov2 = [[0.1, 0], [0, 0.1]]
    class0_1 = np.random.multivariate_normal(mean2, cov2, 10)

    mean3 = [0, 1]
    cov3 = [[0.1, 0], [0, 0.1]]
    class0_2 = np.random.multivariate_normal(mean3, cov3, 10)

    mean4 = [1, 0]
    cov4 = [[0.1, 0], [0, 0.1]]
    class0_3 = np.random.multivariate_normal(mean4, cov4, 10)

    class0 = np.vstack((class0_1, class0_2, class0_3))

    # Combinar as classes
    X_artificial = np.vstack((class1, class0))
    y_artificial = np.array([1]*10 + [0]*30)

    return X_artificial, y_artificial

