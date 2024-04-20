import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from slave import *

# File path to the locally saved dataset
file_path = "dermatology.data"

# Column names for the dataset
column_names = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules', 'follicular_papules',
                'oral_mucosal_involvement', 'knee_and_elbow_involvement', 'scalp_involvement', 'family_history',
                'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis', 'exocytosis',
                'acanthosis', 'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis', 'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer', 'spongiosis', 'saw-tooth_appearance_of_retes', 'follicular_horn_plug', 'erifollicular_parakeratosis', 'inflammatory_mononuclear_inflitrate', 'band-like_infiltrate', 'Age', 'class']

# Read the dataset into a DataFrame
dermatology_data = pd.read_csv(file_path, header=None, names=column_names)

# Convert non-numeric columns to numeric
dermatology_data = dermatology_data.apply(pd.to_numeric, errors='coerce')

# Remove rows and columns with missing values
dermatology_data.dropna(axis=0, how='any', inplace=True)
dermatology_data.dropna(axis=1, how='any', inplace=True)

# Separate features (X) and target (y)
X = dermatology_data.drop(columns=['class']).values
y = dermatology_data['class'].values

# Standardize the features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_standardized = (X - X_mean) / X_std

# Calculate the covariance matrix
cov_matrix = np.cov(X_standardized.T)

# Perform eigendecomposition on the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvectors by eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Choose the number of principal components
total_variance = np.sum(sorted_eigenvalues)
explained_variance_ratio = sorted_eigenvalues / total_variance
cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1

# Project the standardized data onto the new feature space
projection_matrix = sorted_eigenvectors[:, :num_components]
X_pca = X_standardized.dot(projection_matrix)

# Function to plot decision surface
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
    Z = Z.reshape(xx.shape)

    # Plotagem das fronteiras de decisão
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()


# Initialize classifiers
knn_classifier = KNN()
dmc_classifier = DMC()
naive_bayes_classifier = GaussianNB()

# Define the dictionary of classifiers
classifiers = {"KNN": knn_classifier, "DMC": dmc_classifier, "Naive Bayes": naive_bayes_classifier}

# Perform evaluation for each classifier using holdout with 20 runs
for clf_name, clf in classifiers.items():
    print(f"\nClassifier: {clf_name}")
    mean_acc, std_acc, best_idx, best_conf_matrix = holdout_with_20_runs(X_pca, y, clf, test_size=0.3, random_state=42)
    print(f"Mean Accuracy: {mean_acc}")
    print(f"Standard Deviation: {std_acc:.10f}")
    print(f"Best Execution: {best_idx}")
    print("Confusion Matrix:")
    print(best_conf_matrix)
    
    # Plot decision surface
    plot_decision_surface(clf, X_pca, y, f'Decision Surface for {clf_name}')
