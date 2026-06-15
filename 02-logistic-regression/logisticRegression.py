import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLR

np.random.seed(42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


dataset = load_breast_cancer()
feature_index = dataset.feature_names.tolist().index("mean radius")
X = dataset.data[:, feature_index].reshape(-1, 1)
actualY = dataset.target.reshape(-1, 1).astype(float)

# Standardize the feature so gradient descent converges reliably.
X = (X - X.mean()) / X.std()

print(f"Dataset: Breast Cancer Wisconsin ({dataset.target_names[0]} vs {dataset.target_names[1]})")
print(f"Feature: mean radius")


def cost_function(y, y_hat):
    m = y.shape[0]
    # clip to prevent log(0) → -inf
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    return -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def train_test_split(X, y, test_size=0.2):
    numberOfExamples = X.shape[0]
    indices = np.arange(numberOfExamples)
    np.random.shuffle(indices)

    split = int(numberOfExamples * (1 - test_size))

    trainIdx = indices[:split]
    testIdx = indices[split:]

    return X[trainIdx], X[testIdx], y[trainIdx], y[testIdx]


X_train, X_test, y_train, y_test = train_test_split(X, actualY, test_size=0.2)

print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size:  {X_test.shape[0]} samples")


def evaluate(y_true, y_pred_proba, label=""):
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = np.mean(y_pred == y_true)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    print(f"\n{label}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")


class LogisticRegression:
    def __init__(self, learning_rate, epochs) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, actualY):
        m = X.shape[0]
        self.weight = np.random.randn()
        self.bias = np.random.randn()

        print(f"{'Epoch':<10} {'Loss':<15} {'Weight':<12} {'Bias':<12}")
        print("--" * 25)

        for epoch in range(self.epochs + 1):
            z = X * self.weight + self.bias
            predictedY = sigmoid(z)

            error = predictedY - actualY

            dw = (1 / m) * np.sum(error * X)
            db = (1 / m) * np.sum(error)

            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            loss = cost_function(actualY, predictedY)
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(
                    f"{epoch:<10} {loss:<15.6f} {self.weight:<12.6f} {self.bias:<12.6f}"
                )

    def predict_proba(self, X):
        z = X * self.weight + self.bias
        return sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


model = LogisticRegression(learning_rate=0.1, epochs=3000)
model.fit(X_train, y_train)

# --- Evaluate ---

evaluate(y_train, model.predict_proba(X_train), label="Training Set")
print("--" * 25)
evaluate(y_test, model.predict_proba(X_test), label="Test Set")

print("--" * 25)

print(f"Learned weight (w): {model.weight:.4f}")
print(f"Learned bias  (b):  {model.bias:.4f}")

print("--" * 25)

# C=1e10 disables sklearn's default regularization for a fair comparison
sk_model = SklearnLR(C=1e10, solver="lbfgs")
sk_model.fit(X_train, y_train.ravel())

sk_w = sk_model.coef_[0][0]
sk_b = float(np.ravel(sk_model.intercept_)[0])

print(f"{'Model':<20} {'Weight (w)':<15} {'Bias (b)':<15}")
print("--" * 25)
print(f"{'Ours':<20} {model.weight:<15.6f} {model.bias:<15.6f}")
print(f"{'Sklearn':<20} {sk_w:<15.6f} {sk_b:<15.6f}")
print(f"\nDifference w: {abs(model.weight - sk_w):.8f}")
print(
    f"Difference b: {abs((model.bias if model.bias is not None else 0.0) - sk_b):.8f}"
)
