import numpy as np
import pandas as pd
import os
import requests
import zipfile

#############################################
# Definição da classe GaussianNB
# Define o classificador KNN
# DMC
# naive bayes

# Implementação do classificador Naive Bayes
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                "mean": X_c.mean(axis=0),
                "std": X_c.std(axis=0) + 1e-10  # Adicionando um pequeno valor para evitar divisão por zero
            }

    def _pdf(self, X, mean, std):
        return np.exp(-0.5 * ((X - mean) / std) ** 2) / (np.sqrt(2 * np.pi) * std)

    def _predict_class(self, x):
        posteriors = []
        for c in self.classes:
            prior = len(X_train[y_train == c]) / len(X_train)
            likelihood = np.sum(np.log(self._pdf(x, self.parameters[c]["mean"], self.parameters[c]["std"])))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._predict_class(x) for x in X]
        return np.array(y_pred)

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_classes = self.y_train[nearest_neighbors]
            unique_classes, counts = np.unique(nearest_classes, return_counts=True)
            y_pred.append(unique_classes[np.argmax(counts)])
        return np.array(y_pred)

class DMC:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_means = {}
        for c in self.classes:
            self.class_means[c] = np.mean(X_train[y_train == c], axis=0)

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = [np.linalg.norm(x - self.class_means[c]) for c in self.classes]
            y_pred.append(self.classes[np.argmin(distances)])
        return np.array(y_pred)

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

# Inicializar classificadores
knn_classifier = KNN()
dmc_classifier = DMC()
naive_bayes_classifier = GaussianNB()
#
## Definir os classificadores em um dicionário
classifiers = {"KNN": knn_classifier, "DMC": dmc_classifier, "Naive Bayes": naive_bayes_classifier}

def multivariate_gaussian_pdf(x, mean, cov):
    n = len(mean)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    if len(mean.shape) == 1:
        mean = mean.reshape(1, -1)
    diff = x - mean
    exponent = -0.5 * np.sum(np.dot(diff, inv_cov) * diff, axis=1)
    coeff = 1 / ((2 * np.pi) ** (n / 2) * det_cov ** 0.5)
    return coeff * np.exp(exponent)
###################################




# Função para calcular a acurácia
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Função para calcular a matriz de confusão
def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    num_classes = len(classes)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
    return matrix

# Função para carregar o conjunto de dados Iris do repositório UCI
def load_iris_uci():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    iris_data = pd.read_csv(url, names=columns)
    X = iris_data.iloc[:, :-1].values
    y = iris_data.iloc[:, -1].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).values
    return X, y

# Função para separar os dados em conjunto de treinamento e teste
def train_test_split(X, y, test_size=0.3, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    test_samples = int(len(X) * test_size)
    X_train = X[indices[:-test_samples]]
    X_test = X[indices[-test_samples:]]
    y_train = y[indices[:-test_samples]]
    y_test = y[indices[-test_samples:]]
    return X_train, X_test, y_train, y_test





def load_vertebral_column_uci():
    # URL do arquivo ZIP
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip"
    # Caminho local para salvar o arquivo ZIP
    zip_path = "dados\vertebral_column_data.zip"
    # Caminho local para o arquivo de dados extraído
    data_path = "dados\column_3C.dat"

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


def load_dermatology_uci():
    # URL do arquivo ZIP
    url = "https://archive.ics.uci.edu/static/public/33/dermatology.zip"
    # Caminho local para salvar o arquivo ZIP
    zip_path = "dados\dermatology.zip"
    # Caminho local para o arquivo de dados extraído
    data_path = "dados\dermatology.dat"

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
    column_names = [ 'erythema','scaling','definite_borders','itching','koebner_phenomenon','polygonal_papules','follicular_papules'
,'oral_mucosal_involvement','knee_and_elbow_involvement','scalp_involvement','family_history',
'melanin_incontinence','eosinophils_in_the_infiltrate','PNL_infiltrate','fibrosis_of_the_papillary_dermis','exocytosis',
'acanthosis','hyperkeratosis','parakeratosis','clubbing_of_the_rete_ridges','elongation_of_the_rete_ridges','thinning_of_the suprapapillary_epidermis','spongiform_pustule','munro_microabcess','focal_hypergranulosis','disappearance_of_the_granular_layer','vacuolisation_and_damage_of_basal_layer','spongiosis','saw-tooth_appearance_of_retes','follicular_horn_plug','erifollicular_parakeratosis','inflammatory_monoluclear_inflitrate','band-like_infiltrate','Age','class']
    vertebral_data = pd.read_csv(data_path, header=None, sep=' ', names=column_names)
    X = vertebral_data.iloc[:, :-1].values
    y = vertebral_data.iloc[:, -1].values

    return X, y






#### BREAST_CANCER ######

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer = fetch_ucirepo(id=14) 
  
# data (as pandas dataframes) 
X = breast_cancer.data.features 
y = breast_cancer.data.targets 
  
# metadata 
#print(breast_cancer.metadata) 
  
# variable information 
#print(breast_cancer.variables)

##### DERMATOLOGY #####


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
dermatology = fetch_ucirepo(id=33) 
  
# data (as pandas dataframes) 
X = dermatology.data.features 
y = dermatology.data.targets 
  
# metadata 
#print(dermatology.metadata) 
  
# variable information 
#print(dermatology.variables) 


##### COLUMN VERTEBRAL #####

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
vertebral_column = fetch_ucirepo(id=212) 
  
# data (as pandas dataframes) 
X = vertebral_column.data.features 
y = vertebral_column.data.targets 
  
# metadata 
#print(vertebral_column.metadata) 
  
# variable information 
#print(vertebral_column.variables) 

