def predict(X, means, variances, priors):
    predictions = []
    for x in X:
        class_scores = [np.log(priors[i]) - 0.5 * np.sum(np.log(2 * np.pi * variances[i]))
                        - 0.5 * np.sum(((x - means[i][:2]) ** 2) / variances[i][:2]) for i in range(3)]
        predictions.append(np.argmax(class_scores))
    return predictions