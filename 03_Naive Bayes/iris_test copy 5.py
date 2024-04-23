import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import norm, multivariate_normal  # Importando multivariate_normal
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data  
y = iris.target  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def compute_likelihood(X, mean, cov):
    n = X.shape[0]
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    exponent = -0.5 * np.sum(np.dot(X - mean, inv_cov) * (X - mean), axis=1)
    likelihood = (2 * np.pi) ** (-n / 2) * det_cov ** (-0.5) * np.exp(exponent)
    return likelihood

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = {}
        self.mean = {}
        self.cov = {}

        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = len(X_c) / len(X)
            self.mean[c] = np.mean(X_c, axis=0)
            self.cov[c] = np.cov(X_c, rowvar=False)

    def predict(self, X):
        predictions = []
        posteriors = []

        for x in X:
            posteriors_per_instance = []

            for c in self.classes:
                likelihood = compute_likelihood(np.atleast_2d(x), self.mean[c], self.cov[c])
                posterior = self.class_probs[c] * likelihood
                posteriors_per_instance.append(posterior)

            predicted_class = self.classes[np.argmax(posteriors_per_instance)]
            predictions.append(predicted_class)
            posteriors.append(posteriors_per_instance)

        class_posteriors = {c: [] for c in self.classes}
        
        for posterior in posteriors:
            for i, p in enumerate(posterior):
                class_posteriors[self.classes[i]].append(p)

        mean_class_posteriors = {c: np.mean(class_posteriors[c]) for c in self.classes}
        return predictions, mean_class_posteriors
    
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    num_classes = len(classes)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))
    return matrix

def train_test_split(X, y, test_size=0.2, random_state=None):
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

def perform_realizations(classifier, X, y):
    accuracies = []
    class_posteriors_all = []

    for _ in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=_)
        classifier.fit(X_train, y_train)
        y_pred, class_posteriors = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        class_posteriors_all.append(class_posteriors)

    accuracies = np.array(accuracies)
    mean_accuracy = np.mean(accuracies)
    std_dev_accuracy = np.std(accuracies)
    best_realization_idx = np.argmin(np.abs(accuracies - mean_accuracy))

    # Retornando apenas os resultados da melhor realização
    best_class_posteriors = class_posteriors_all[best_realization_idx]
    
    # Imprimindo as informações
    print("Dataset: Iris")
    print(f"Melhor acurácia: {mean_accuracy}")
    print(f"Desvio padrão da acurácia: {std_dev_accuracy}")
    print(f"Melhor realização (índice): {best_realization_idx}")
    print("Matriz de confusão da melhor realização:")
    print(confusion_matrix(y_test, y_pred))

    return mean_accuracy, std_dev_accuracy, best_realization_idx, best_class_posteriors

def plot_decision_surface(clf, X_train, y_train, title):
    # Escolha dos dois primeiros atributos
    X_train = X_train[:, :2]
    
    # Treinar o classificador com os dois primeiros atributos
    clf.fit(X_train, y_train)

    # Criação da grade de pontos
    h = .02  # tamanho do passo na grade
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predição para cada ponto na grade
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])[0]  # Obter apenas as previsões e não as probabilidades

    # Converter Z para um array numpy
    Z = np.array(Z)
    
    # Colocando o resultado na mesma forma que a grade
    Z = Z.reshape(xx.shape)

    # Plotagem das fronteiras de decisão
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(title)
    plt.show()

def plot_gaussian_over_data(X_train, y_train, X_test, y_test, title):
    plt.figure(figsize=(12, 8))

    for i, class_name in enumerate(np.unique(y_train)):
        X_class_train = X_train[y_train == class_name]
        X_class_test = X_test[y_test == class_name]

        # Plot Gaussian distribution for training data
        mean = np.mean(X_class_train, axis=0)
        cov = np.cov(X_class_train, rowvar=False)
        plt.scatter(X_class_train[:, 0], X_class_train[:, 1], label=f'Train Class {class_name}')
        plot_gaussian_2d(mean, cov)

        # Plot Gaussian distribution for test data
        plt.scatter(X_class_test[:, 0], X_class_test[:, 1], label=f'Test Class {class_name}')
        plt.scatter(mean[0], mean[1], marker='o', color='red', label=f'Mean Test Class {class_name}')

    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(title)
    plt.legend()
    plt.show()

def plot_gaussian_2d(mean, cov, n_std=1):
    x, y = np.meshgrid(np.linspace(mean[0] - 3 * np.sqrt(cov[0, 0]), mean[0] + 3 * np.sqrt(cov[0, 0]), 100),
                       np.linspace(mean[1] - 3 * np.sqrt(cov[1, 1]), mean[1] + 3 * np.sqrt(cov[1, 1]), 100))
    pos = np.dstack((x, y)).reshape(-1, 2)
    rv = multivariate_normal(mean, cov)
    plt.contour(x, y, rv.pdf(pos).reshape(100, 100), levels=[rv.pdf(mean.reshape(1, -1)) * np.exp(-0.5)], colors='black', alpha=0.5)

# Instanciando e treinando o classificador Naive Bayes
nb_classifier = NaiveBayesClassifier()

# Obtendo os resultados da melhor realização
mean_acc, std_dev_acc, best_realization_idx, best_class_posteriors = perform_realizations(nb_classifier, X, y)

# Plotando as probabilidades médias a posteriori para cada classe
plt.figure(figsize=(10, 6))
plt.bar(range(len(best_class_posteriors)), list(best_class_posteriors.values()))
plt.xlabel('Classes')
plt.ylabel('Probabilidade Média a Posteriori')
plt.title('Probabilidade Média a Posteriori para Cada Classe')
plt.xticks(range(len(best_class_posteriors)), iris.target_names)
plt.show()

# Plotando a superfície de decisão
plot_decision_surface(nb_classifier, X_train, y_train, "Superfície de Decisão - Naive Bayes")

# Plotando as gaussianas sobre os dados para cada uma das classes
plot_gaussian_over_data(X_train, y_train, X_test, y_test, "Distribuição Gaussiana sobre os Dados")
