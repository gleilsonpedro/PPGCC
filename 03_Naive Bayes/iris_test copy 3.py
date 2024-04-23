import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Função para carregar o conjunto de dados Iris
def load_iris():
    from sklearn import datasets
    iris = datasets.load_iris()
    return iris.data, iris.target

# Dividir os dados em conjuntos de treinamento e teste
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

# Função para calcular a verossimilhança de uma amostra para uma distribuição normal
def compute_likelihood(x, mean, std):
    likelihood = np.prod(norm.pdf(x, mean, std))
    return likelihood

# Classe do classificador Naive Bayes
class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = {}
        self.mean = {}
        self.std = {}

        for c in self.classes:
            X_c = X[y == c]
            self.class_probs[c] = len(X_c) / len(X)
            self.mean[c] = np.mean(X_c, axis=0)
            self.std[c] = np.std(X_c, axis=0)

    def predict(self, X):
        predictions = []

        for x in X:
            posteriors_per_instance = []

            for c in self.classes:
                likelihood = compute_likelihood(x, self.mean[c], self.std[c])
                posterior = self.class_probs[c] * likelihood
                posteriors_per_instance.append(posterior)

            predicted_class = self.classes[np.argmax(posteriors_per_instance)]
            predictions.append(predicted_class)

        return predictions

    def class_posteriors(self, X):
        class_posteriors = {c: [] for c in self.classes}

        for x in X:
            posteriors_per_instance = []

            for c in self.classes:
                likelihood = compute_likelihood(x, self.mean[c], self.std[c])
                posterior = self.class_probs[c] * likelihood
                posteriors_per_instance.append(posterior)

            for i, c in enumerate(self.classes):
                class_posteriors[c].append(posteriors_per_instance[i])

        mean_class_posteriors = {c: np.mean(class_posteriors[c]) for c in class_posteriors}
        return mean_class_posteriors

# Função para calcular as probabilidades médias a posteriori para cada classe
def plot_posteriors(class_posteriors):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_posteriors)), class_posteriors.values())
    plt.xlabel('Classes')
    plt.ylabel('Probabilidade Média a Posteriori')
    plt.title('Probabilidade Média a Posteriori para Cada Classe')
    plt.xticks(range(len(class_posteriors)), class_posteriors.keys())
    plt.show()

# Função para plotar a superfície de decisão
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
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Colocando o resultado na mesma forma que a grade
    Z = np.array(Z).reshape(xx.shape)

    # Plotagem das fronteiras de decisão
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

# Carregar os dados
X, y = load_iris()

# Instanciar o classificador Naive Bayes
nb_classifier = NaiveBayesClassifier()

# Executar as iterações e obter os resultados da melhor realização
mean_acc, std_dev_acc, best_realization_idx, _ = perform_realizations(nb_classifier, X, y)

# Calcular as probabilidades médias a posteriori para cada classe na melhor realização
best_class_posteriors = nb_classifier.class_posteriors(X)

# Plotar as probabilidades médias a posteriori para cada classe
plot_posteriors(best_class_posteriors)

# Plotar a superfície de decisão
plot_decision_surface(nb_classifier, X, y, "Superfície de Decisão - Naive Bayes")
