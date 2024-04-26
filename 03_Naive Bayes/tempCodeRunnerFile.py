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