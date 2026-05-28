import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR

np.random.seed(42)


def generateSynData(rows):
    noise = np.random.randn(rows, 1)
    X = 2 * np.random.rand(rows, 1)
    y = 4 + (3 * X) + noise
    # output = the bias + the weight times input features + noise

    return X, y


X, actualY = generateSynData(1000)


def cost_function(numberOfExamples, error):
    return (1 / (2 * numberOfExamples)) * np.sum(error**2)


def train_test_split(X, actualY, test_size=0.2):
    numberOfexamples = X.shape[0]
    indices = np.arange(numberOfexamples)
    np.random.shuffle(indices)

    split = int(numberOfexamples * (1 - test_size))

    trainIdx = indices[:split]
    testIdx = indices[split:]

    return X[trainIdx], X[testIdx], actualY[trainIdx], actualY[testIdx]


X_train, X_test, y_train, y_test = train_test_split(X, actualY, test_size=0.2)

print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size:  {X_test.shape[0]} samples")


def evaluate(y_true, y_pred, label=""):
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


class LinearRegression:
    def __init__(self, learning_rate, epochs) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, actualY):
        numberOfExamples = X.shape[0]
        self.weight = np.random.randn()
        self.bias = np.random.randn()

        print(f"{'Epoch':<10} {'Loss':<15} {'Weight':<12} {'Bias':<12}")
        print("--" * 25)

        for epoch in range(self.epochs + 1):
            predictedY = X * self.weight + self.bias

            error = predictedY - actualY

            dw = (1 / numberOfExamples) * np.sum(error * X)
            db = (1 / numberOfExamples) * np.sum(error)

            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            loss = cost_function(numberOfExamples, error)
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(
                    f"{epoch:<10} {loss:<15.6f} {self.weight:<12.6f} {self.bias:<12.6f}"
                )

    def predict(self, X):
        return X * self.weight + self.bias


model = LinearRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# --- Evaluate ---

evaluate(y_train, model.predict(X_train), label="Training Set")
print("--" * 25)
evaluate(y_test, model.predict(X_test), label="Test Set")

print("--" * 25)

print(f"Learned weight (w): {model.weight:.4f}")
print(f"Learned bias  (b):  {model.bias:.4f}")

print("--" * 25)


sk_model = SklearnLR()
sk_model.fit(X, actualY)

sk_w = sk_model.coef_[0][0]
sk_b = sk_model.intercept_[0]

print(f"{'Model':<20} {'Weight (w)':<15} {'Bias (b)':<15}")
print("--" * 25)
print(f"{'Ours':<20} {model.weight:<15.6f} {model.bias:<15.6f}")
print(f"{'Sklearn':<20} {sk_w:<15.6f} {sk_b:<15.6f}")
print(f"{'True values':<20} {'3.000000':<15} {'4.000000':<15}")
print(f"\nDifference w: {abs(model.weight - sk_w):.8f}")
print(f"Difference b: {abs(model.bias - sk_b):.8f}")
