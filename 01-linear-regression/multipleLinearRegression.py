import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR

np.random.seed(42)

numberOfExamples = 1000

rawX = np.random.rand(numberOfExamples, 3)
actualWeights = np.array([2.0, 2.5, -1.0])
bias = 5.0
noise = np.random.randn(numberOfExamples) * 0.5

actualY = rawX @ actualWeights + bias + noise
actualY = actualY.reshape(-1, 1)


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
        m, n = X.shape  # m=160 examples, n=4 parameters
        self.weight = np.zeros((n, 1))  # [[0], [0], [0], [0]]

        print(
            f"\n{'Epoch':<10} {'Loss':<15} {'w1':<10} {'w2':<10} {'w3':<10} {'b':<10}"
        )
        print("-" * 65)

        for epoch in range(self.epochs + 1):
            predictedY = X @ self.weight  # (160,1) = (160,4)@(4,1)
            error = predictedY - actualY  # (160,1)
            grad = (1 / m) * X.T @ error  # (4,1)  = (4,160)@(160,1)
            self.weight = self.weight - self.learning_rate * grad

            loss = (1 / (2 * m)) * np.sum(error**2)
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                w = self.weight.flatten()
                print(
                    f"{epoch:<10} {loss:<15.6f} {w[0]:<10.4f} {w[1]:<10.4f} {w[2]:<10.4f} {w[3]:<10.4f}"
                )

    def predict(self, X):
        return X @ self.weight


# --- Train ---
model = MutlipleLinearRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# --- Learned vs True ---
assert model.weight is not None, "Model must be fitted before accessing weights"
w = model.weight.flatten()
print(f"\n{'Parameter':<12} {'Learned':<12} {'True':<12}")
print("-" * 36)
print(f"{'w1':<12} {w[0]:<12.4f} {actualWeights[0]}")
print(f"{'w2':<12} {w[1]:<12.4f} {actualWeights[1]}")
print(f"{'w3':<12} {w[2]:<12.4f} {actualWeights[2]}")
print(f"{'b':<12} {w[3]:<12.4f} {bias}")

# --- Evaluate ---
evaluate(y_train, model.predict(X_train), "Training Set")
evaluate(y_test, model.predict(X_test), "Test Set")

# --- Sklearn Comparison ---
sk = SklearnLR()
sk.fit(X_train[:, :-1], y_train)
evaluate(y_test, sk.predict(X_test[:, :-1]), "Sklearn Test Set")
