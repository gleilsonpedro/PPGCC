import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Função para calcular a média e a matriz de covariância de cada classe
def calculate_class_statistics(X, y):
    class_means = []
    class_covariances = []
    for i in range(3):
        X_class = X[y == i]
        class_means.append(np.mean(X_class, axis=0))
        class_covariances.append(np.cov(X_class.T))
    return class_means, class_covariances

# Função para calcular a função discriminante quadrática
def quadratic_discriminant_function(x, class_mean, class_covariance):
    inv_covariance = np.linalg.inv(class_covariance)
    det_covariance = np.linalg.det(class_covariance)
    constant_term = -0.5 * np.log(det_covariance)
    quadratic_term = -0.5 * np.dot(np.dot((x - class_mean).T, inv_covariance), (x - class_mean))
    return quadratic_term + constant_term

# Função para fazer previsões usando a função discriminante quadrática
def predict_quadratic(X, class_means, class_covariances, class_priors):
    predictions = []
    for x in X:
        discriminant_values = [quadratic_discriminant_function(x, class_means[i], class_covariances[i]) for i in range(3)]
        predictions.append(np.argmax(discriminant_values))
    return predictions

# Gerar conjunto de dados artificial
def generate_artificial_dataset():
    np.random.seed(42)
    class0 = np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], 8)
    class1 = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], 8)
    class2 = np.random.multivariate_normal([3, 3], [[0.1, 0], [0, 0.1]], 7)
    X = np.vstack([class0, class1, class2])
    y = np.array([0]*8 + [1]*8 + [2]*7)
    return X, y

# Função para calcular métricas de classificação
def calculate_metrics(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    confusion_matrix = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true, pred] += 1
    return accuracy, confusion_matrix

# Gerar dataset artificial
X, y = generate_artificial_dataset()

# Holdout com 20 realizações
n_realizations = 20
accuracies = []
confusion_matrices = []
for _ in range(n_realizations):
    # Dividir conjunto de dados em treino e teste
    np.random.shuffle(X)
    np.random.shuffle(y)
    X_train, X_test = X[:15], X[15:]
    y_train, y_test = y[:15], y[15:]

    # Calcular priors, médias e covariâncias de classe
    class_priors = np.bincount(y_train) / len(y_train)
    class_means, class_covariances = calculate_class_statistics(X_train, y_train)

    # Fazer previsões no conjunto de teste usando a função discriminante quadrática
    y_pred = predict_quadratic(X_test, class_means, class_covariances, class_priors)

    # Calcular métricas de classificação
    accuracy, confusion_matrix = calculate_metrics(y_test, y_pred)
    accuracies.append(accuracy)
    confusion_matrices.append(confusion_matrix)

# Calcular média e desvio padrão das acurácias
mean_accuracy = np.mean(accuracies)
std_dev_accuracy = np.std(accuracies)

# Encontrar a melhor realização com base na acurácia mais próxima da média
best_realization_index = np.argmin(np.abs(np.array(accuracies) - mean_accuracy))
best_accuracy = accuracies[best_realization_index]
best_confusion_matrix = confusion_matrices[best_realization_index]

# Imprimir resultados
print("Dataset: Artificial\n")
print(f"Melhor Realização: {best_realization_index}")
print(f"Acurácia: {best_accuracy:.4f}")
print(f"Desvio Padrão: {std_dev_accuracy:.4f}")
print("Matriz de Confusão:")
print(best_confusion_matrix)

# Imprimir matrizes de covariância
print("\nMatriz de Covariância Completa:")
for i, covariance in enumerate(class_covariances):
    print(f"Classe {i}:")
    print(covariance)

print("\nMatriz de Covariância Diagonal:")
for i, covariance in enumerate(class_covariances):
    print(f"Classe {i}:")
    print(np.diag(np.diag(covariance)))

print("\nMatriz de Covariância Equal:")
equal_covariance = np.mean(np.array(class_covariances), axis=0)
for i, _ in enumerate(class_covariances):
    print(f"Classe {i}:")
    print(equal_covariance)

# Plotar superfície de decisão
def plot_decision_surface(X, y, class_means, class_covariances):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict_quadratic(np.c_[xx.ravel(), yy.ravel()], class_means, class_covariances, class_priors)
    Z = np.array(Z).reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', s=20, label='Training Data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test Data', edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Artificial Dataset - Decision Surface')
    plt.legend()

# Plotar superfície de decisão para a melhor realização
plt.figure(figsize=(8, 6))
plot_decision_surface(X, y, class_means, class_covariances)

plt.show()
