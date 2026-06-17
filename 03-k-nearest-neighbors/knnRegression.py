import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor as SklearnKNNRegressor

np.random.seed(42)


def loadDataset():
    diabetes = load_diabetes()

    bmiIndex = diabetes.feature_names.index("bmi")

    X = diabetes.data[:, bmiIndex].reshape(-1, 1)
    y = diabetes.target.reshape(-1, 1)

    return X, y


X, actualY = loadDataset()


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

    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_validation_scaled = (X_validation - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_validation_scaled, X_test_scaled, mean, std


X_train, X_validation, X_test, mean, std = standardize_data(
    X_train, X_validation, X_test
)


def evaluate(y_true, y_pred, label=""):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    errors = y_pred - y_true

    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    r2 = 1 - (np.sum(errors**2) / np.sum((y_true - np.mean(y_true)) ** 2))

    print(f"\n{label}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")


def rmse_score(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    errors = y_pred - y_true
    rmse = np.sqrt(np.mean(errors**2))

    return rmse


class KNNRegressor:
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

        nearestTargets = self.y_train[nearestNeighborsIndexes]
        nearestDistances = distances[nearestNeighborsIndexes]

        if not self.weighted:
            predictedValue = np.mean(nearestTargets)

            return predictedValue

        epsilon = 1e-8
        weights = 1 / (nearestDistances + epsilon)

        predictedValue = np.sum(weights * nearestTargets) / np.sum(weights)

        return predictedValue

    def predict(self, X):
        predictedY = np.array([self.predict_one(x) for x in X])

        return predictedY


def choose_best_k(k_values, X_train, y_train, X_validation, y_validation):
    bestK = None
    bestValidationRMSE = float("inf")

    print(f"\n{'k':<10} {'Train RMSE':<20} {'Validation RMSE':<20}")
    print("--" * 35)

    for k in k_values:
        model = KNNRegressor(k=k)
        model.fit(X_train, y_train)

        trainPredictions = model.predict(X_train)
        validationPredictions = model.predict(X_validation)

        trainRMSE = rmse_score(y_train, trainPredictions)
        validationRMSE = rmse_score(y_validation, validationPredictions)

        print(f"{k:<10} {trainRMSE:<20.4f} {validationRMSE:<20.4f}")

        if validationRMSE < bestValidationRMSE:
            bestValidationRMSE = validationRMSE
            bestK = k

    return bestK, bestValidationRMSE


k_values = [1, 3, 5, 7, 9, 11]

bestK, bestValidationRMSE = choose_best_k(
    k_values,
    X_train,
    y_train,
    X_validation,
    y_validation,
)

print(f"\nBest k: {bestK}")
print(f"Best validation RMSE: {bestValidationRMSE:.4f}")


normalModel = KNNRegressor(k=bestK, weighted=False)
normalModel.fit(X_train, y_train)

weightedModel = KNNRegressor(k=bestK, weighted=True)
weightedModel.fit(X_train, y_train)

sk_model = SklearnKNNRegressor(n_neighbors=bestK)
sk_model.fit(X_train, y_train.reshape(-1))

print("\nNormal KNN Regression")
print("--" * 25)
evaluate(y_train, normalModel.predict(X_train), label="Training Set")
evaluate(y_validation, normalModel.predict(X_validation), label="Validation Set")
evaluate(y_test, normalModel.predict(X_test), label="Test Set")

print("\nWeighted KNN Regression")
print("--" * 25)
evaluate(y_train, weightedModel.predict(X_train), label="Training Set")
evaluate(y_validation, weightedModel.predict(X_validation), label="Validation Set")
evaluate(y_test, weightedModel.predict(X_test), label="Test Set")

print("\nSklearn KNN Regression")
print("--" * 25)
evaluate(y_train, sk_model.predict(X_train), label="Training Set")
evaluate(y_validation, sk_model.predict(X_validation), label="Validation Set")
evaluate(y_test, sk_model.predict(X_test), label="Test Set")

print("--" * 25)


our_pred = normalModel.predict(X_test)
sk_pred = sk_model.predict(X_test)

our_rmse = rmse_score(y_test, our_pred)
sk_rmse = rmse_score(y_test, sk_pred)

print(f"{'Model':<20} {'Test RMSE':<15}")
print("--" * 25)
print(f"{'Ours':<20} {our_rmse:<15.6f}")
print(f"{'Sklearn':<20} {sk_rmse:<15.6f}")

print(f"\nSame predictions as sklearn: {np.allclose(our_pred, sk_pred)}")
