import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data  
y = iris.target  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def compute_likelihood(X, mean, std):
    likelihood = np.prod(norm.pdf(X, mean, std))
    return likelihood

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
        posteriors = []

        for x in X:
            posteriors_per_instance = []

            for c in self.classes:
                likelihood = compute_likelihood(x, self.mean[c], self.std[c])
                posterior = self.class_probs[c] * likelihood
                posteriors_per_instance.append(posterior)

            predicted_class = self.classes[np.argmax(posteriors_per_instance)]
            predictions.append(predicted_class)
            posteriors.append(posteriors_per_instance)

        class_posteriors = {c: [] for c in self.classes}
        
        for posterior in posteriors:
            for i, p in enumerate(posterior):
                class_posteriors[self.classes[i]].append(p)

        mean_class_posteriors = {c: np.mean(class_posteriors[c], axis=0) for c in self.classes}
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

# Instanciando e treinando o classificador Naive Bayes
nb_classifier = NaiveBayesClassifier()

# Obtendo os resultados da melhor realização
mean_acc, std_dev_acc, best_realization_idx, best_class_posteriors = perform_realizations(nb_classifier, X, y)

# Plotando as probabilidades médias a posteriori para cada classe
plt.figure(figsize=(10, 6))
plt.bar(range(len(best_class_posteriors)), best_class_posteriors.values())
plt.xlabel('Classes')
plt.ylabel('Probabilidade Média a Posteriori')
plt.title('Probabilidade Média a Posteriori para Cada Classe')
plt.xticks(range(len(best_class_posteriors)), iris.target_names)
plt.show()
