import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.datasets import load_diabetes

np.random.seed(42)


def loadDataset():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target.reshape(-1, 1)

    return X, y, diabetes.feature_names


rawX, actualY, feature_names = loadDataset()


def add_bias(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack([X, ones])


X = add_bias(rawX)


# --- Train/Test Split ---
def train_test_split(X, y, test_size=0.2):
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    split = int(m * (1 - test_size))  # 160

    return (
        X[indices[:split]],
        X[indices[split:]],
        y[indices[:split]],
        y[indices[split:]],
    )


X_train, X_test, y_train, y_test = train_test_split(X, actualY)

print(f"Train: {X_train.shape}  Test: {X_test.shape}")


# --- Metrics ---
def evaluate(actualY, predictedY, label=""):
    errors = predictedY - actualY
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    r2 = 1 - (np.sum(errors**2) / np.sum((actualY - np.mean(actualY)) ** 2))

    print(f"\n{label}")
    print("-" * 30)
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")


class MutlipleLinearRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = None
        self.loss_history = []

    def fit(self, X, actualY):
        m, n = X.shape
        self.weight = np.zeros((n, 1))

        print(f"\n{'Epoch':<10} {'Loss':<15} {'Weight norm':<15} {'Bias':<10}")
        print("-" * 55)

        for epoch in range(self.epochs + 1):
            predictedY = X @ self.weight
            error = predictedY - actualY
            gradient = (1 / m) * X.T @ error
            self.weight = self.weight - self.learning_rate * gradient

            loss = (1 / (2 * m)) * np.sum(error**2)
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                w = self.weight.flatten()
                print(
                    f"{epoch:<10} {loss:<15.6f} {np.linalg.norm(w[:-1]):<15.4f} {w[-1]:<10.4f}"
                )

    def predict(self, X):
        return X @ self.weight


# --- Train ---
model = MutlipleLinearRegression(learning_rate=1.0, epochs=10000)
model.fit(X_train, y_train)

# --- Evaluate ---
evaluate(y_train, model.predict(X_train), "Training Set")
evaluate(y_test, model.predict(X_test), "Testing Set")

# --- Sklearn Comparison ---
sk = SklearnLR()
sk.fit(X_train[:, :-1], y_train)
evaluate(y_train, sk.predict(X_train[:, :-1]), "Sklearn Training Set")
evaluate(y_test, sk.predict(X_test[:, :-1]), "Sklearn Testing Set")

# --- Learned vs Sklearn ---
assert model.weight is not None, "Model must be fitted before accessing weights"
w = model.weight.flatten()
sk_w = sk.coef_.flatten()
sk_b = sk.intercept_[0]

print(f"\n{'Parameter':<12} {'Ours':<12} {'Sklearn':<12} {'Difference':<12}")
print("-" * 52)

for name, ours, sklearn_weight in zip(feature_names, w[:-1], sk_w):
    print(
        f"{name:<12} {ours:<12.4f} {sklearn_weight:<12.4f} {abs(ours - sklearn_weight):<12.4f}"
    )

print(f"{'bias':<12} {w[-1]:<12.4f} {sk_b:<12.4f} {abs(w[-1] - sk_b):<12.4f}")
