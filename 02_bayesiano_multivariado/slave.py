import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

# Function to perform holdout with 20 runs
def holdout_with_20_runs(X, y, clf, test_size=0.3, random_state=None):
    accs = []
    conf_matrices = []
    for _ in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        acc, conf_matrix = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
        accs.append(acc)
        conf_matrices.append(conf_matrix)
    accs = np.array(accs)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    best_idx = np.argmax(accs)
    best_conf_matrix = conf_matrices[best_idx]
    return mean_acc, std_acc, best_idx + 1, best_conf_matrix


# Function to calculate accuracy
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Function to calculate confusion matrix
def confusion_matrix(y_true, y_pred, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for i in range(len(labels)):
        for j in range(len(labels)):
            matrix[i, j] = np.sum((y_true == labels[i]) & (y_pred == labels[j]))
    return matrix

def train_test_split(X, y, test_size=0.3, random_state=None):
    if random_state:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test_samples = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test_samples]
    train_indices = indices[n_test_samples:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test




# Define a função para carregar o conjunto de dados Dermatology
def load_dermatology_uci():
    # Obtém o diretório atual do script
    script_dir = os.path.dirname(__file__)

    # Caminho para o arquivo 'dermatology.data' dentro da pasta 'dados'
    data_file = os.path.join(script_dir, "dados", "dermatology.data")


    column_names = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules', 'follicular_papules',
                    'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement', 'family_history',
                    'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis', 'exocytosis',
                    'acanthosis', 'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis', 'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer', 'spongiosis', 'saw-tooth_appearance_of_retes', 'follicular_horn_plug', 'erifollicular_parakeratosis', 'inflammatory_mononuclear_inflitrate', 'band-like_infiltrate', 'Age', 'class']
    
    # Carrega os dados
    dermatology_data = pd.read_csv(data_file, header=None, names=column_names)
   



    # Converter colunas não numéricas para numéricas
    dermatology_data = dermatology_data.apply(pd.to_numeric, errors='coerce')
    
    # Remover linhas e colunas com valores ausentes
    dermatology_data.dropna(axis=0, how='any', inplace=True)
    dermatology_data.dropna(axis=1, how='any', inplace=True)
    
    # Separar características (X) e alvo (y)
    X = dermatology_data.drop(columns=['class']).values
    y = dermatology_data['class'].values
    
    return X, y

# Carregar dados Dermatology
X, y = load_dermatology_uci()
#
## Standardize the features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_standardized = (X - X_mean) / X_std

# Define a função para avaliar o classificador
def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    return acc, conf_matrix

# Define a função para realizar holdout com 20 execuções
def holdout_with_20_runs(X, y, clf, test_size=0.3, random_state=None):
    accs = []
    conf_matrices = []
    for _ in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        acc, conf_matrix = evaluate_classifier(clf, X_train, X_test, y_train, y_test)
        accs.append(acc)
        conf_matrices.append(conf_matrix)
    accs = np.array(accs)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    best_idx = np.argmax(accs)
    best_conf_matrix = conf_matrices[best_idx]
    return mean_acc, std_acc, best_idx + 1, best_conf_matrix

# Define o classificador KNN
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

# Executar avaliação para cada classificador usando holdout com 20 execuções
for clf_name, clf in classifiers.items():
    print(f"\nClassifier: {clf_name}")
    mean_acc, std_acc, best_idx, best_conf_matrix = holdout_with_20_runs(X_standardized, y, clf, test_size=0.3, random_state=42)
    print(f"Mean Accuracy: {mean_acc}")
    print(f"Standard Deviation: {std_acc:.10f}")
    print(f"Best Execution: {best_idx}")
    print("Confusion Matrix:")
    print(best_conf_matrix)
