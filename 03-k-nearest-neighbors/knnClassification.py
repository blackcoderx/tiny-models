import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

np.random.seed(42)


def loadDataset():
    iris = load_iris()

    X = iris.data
    y = iris.target

    classNames = iris.target_names

    return X, y, classNames


X, actualY, classNames = loadDataset()


def train_validation_test_split(X, actualY, train_size=0.6, validation_size=0.2):
    numberOfexamples = X.shape[0]

    indices = np.arange(numberOfexamples)
    np.random.shuffle(indices)

    trainEnd = int(numberOfexamples * train_size)
    validationEnd = int(numberOfexamples * (train_size + validation_size))

    trainIdx = indices[:trainEnd]
    validationIdx = indices[trainEnd:validationEnd]
    testIdx = indices[validationEnd:]

    return (
        X[trainIdx],
        X[validationIdx],
        X[testIdx],
        actualY[trainIdx],
        actualY[validationIdx],
        actualY[testIdx],
    )


X_train, X_validation, X_test, y_train, y_validation, y_test = (
    train_validation_test_split(X, actualY)
)

print(f"Train size:      {X_train.shape[0]} samples")
print(f"Validation size: {X_validation.shape[0]} samples")
print(f"Test size:       {X_test.shape[0]} samples")


def standardize_data(X_train, X_validation, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_validation_scaled = (X_validation - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_validation_scaled, X_test_scaled, mean, std


X_train, X_validation, X_test, mean, std = standardize_data(
    X_train, X_validation, X_test
)


def evaluate(y_true, y_pred, label=""):
    accuracy = np.mean(y_true == y_pred)

    print(f"\n{label}")
    print(f"  Accuracy: {accuracy:.4f}")


def choose_best_k(k_values, X_train, y_train, X_validation, y_validation):
    bestK = None
    bestValidationAccuracy = 0

    print(f"\n{'k':<10} {'Train Accuracy':<20} {'Validation Accuracy':<20}")
    print("--" * 35)

    for k in k_values:
        model = KNNClassifier(k=k)
        model.fit(X_train, y_train)

        trainPredictions = model.predict(X_train)
        validationPredictions = model.predict(X_validation)

        trainAccuracy = np.mean(y_train == trainPredictions)
        validationAccuracy = np.mean(y_validation == validationPredictions)

        print(f"{k:<10} {trainAccuracy:<20.4f} {validationAccuracy:<20.4f}")

        if validationAccuracy > bestValidationAccuracy:
            bestValidationAccuracy = validationAccuracy
            bestK = k

    return bestK, bestValidationAccuracy


class KNNClassifier:
    def __init__(self, k=3, weighted=False) -> None:
        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X, actualY):
        self.X_train = X
        self.y_train = actualY.reshape(-1)

    def euclidean_distance(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        return distances

    def predict_one(self, x):
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Model has not been fitted yet. Call fit() before predict()."
            )

        distances = self.euclidean_distance(x)

        nearestNeighborsIndexes = np.argsort(distances)[: self.k]

        nearestLabels = self.y_train[nearestNeighborsIndexes]
        nearestDistances = distances[nearestNeighborsIndexes]

        if not self.weighted:
            labels, counts = np.unique(nearestLabels, return_counts=True)
            predictedLabel = labels[np.argmax(counts)]

            return predictedLabel

        labels = np.unique(nearestLabels)

        epsilon = 1e-8
        weights = 1 / (nearestDistances + epsilon)

        weightedVotes = []

        for label in labels:
            vote = np.sum(weights[nearestLabels == label])
            weightedVotes.append(vote)

        weightedVotes = np.array(weightedVotes)

        predictedLabel = labels[np.argmax(weightedVotes)]

        return predictedLabel

    def predict(self, X):
        predictedY = np.array([self.predict_one(x) for x in X])

        return predictedY


k_values = [1, 3, 5, 7, 9, 11]

bestk, bestacc = choose_best_k(k_values, X_train, y_train, X_validation, y_validation)

normalModel = KNNClassifier(k=bestk, weighted=False)
normalModel.fit(X_train, y_train)

weightedModel = KNNClassifier(k=bestk, weighted=True)
weightedModel.fit(X_train, y_train)

sk_model = SklearnKNN(n_neighbors=5)
sk_model.fit(X_train, y_train)

print("\nNormal KNN Classification")
print("--" * 25)
evaluate(y_train, normalModel.predict(X_train), label="Training Set")
evaluate(y_validation, normalModel.predict(X_validation), label="Validation Set")
evaluate(y_test, normalModel.predict(X_test), label="Test Set")

print("\nWeighted KNN Classification")
print("--" * 25)
evaluate(y_train, weightedModel.predict(X_train), label="Training Set")
evaluate(y_validation, weightedModel.predict(X_validation), label="Validation Set")
evaluate(y_test, weightedModel.predict(X_test), label="Test Set")

print("\nSkLearn KNN Classification")
print("--" * 25)
evaluate(y_train, sk_model.predict(X_train), label="Training Set")
evaluate(y_validation, sk_model.predict(X_validation), label="Validation Set")
evaluate(y_test, sk_model.predict(X_test), label="Test Set")


normalPrediction = normalModel.predict(X_test)
weightedPrediction = weightedModel.predict(X_test)
sk_pred = sk_model.predict(X_test)


print(f"\nSame predictions as sklearn(normal): {np.all(normalPrediction == sk_pred)}")
print(
    f"\nSame predictions as sklearn(weighted): {np.all(weightedPrediction == sk_pred)}"
)
