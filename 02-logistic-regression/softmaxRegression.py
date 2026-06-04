import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target

    return X, y


X, actualY = load_data()


def train_test_split(X, actualY, test_size=0.2):
    numberOfExamples = X.shape[0]
    indices = np.arange(numberOfExamples)
    np.random.shuffle(indices)

    split = int(numberOfExamples * (1 - test_size))

    trainIdx = indices[:split]
    testIdx = indices[split:]

    return X[trainIdx], X[testIdx], actualY[trainIdx], actualY[testIdx]


def one_hot_encode(y, numberOfClasses):
    numberOfExamples = y.shape[0]

    oneHotY = np.zeros((numberOfExamples, numberOfClasses))

    oneHotY[np.arange(numberOfExamples), y] = 1

    return oneHotY


def softmax(scores):
    scores = scores - np.max(scores, axis=1, keepdims=True)

    expScores = np.exp(scores)

    probabilities = expScores / np.sum(expScores, axis=1, keepdims=True)

    return probabilities


def cost_function(numberOfExamples, actualYOneHot, predictedProbabilities):
    epsilon = 1e-15

    predictedProbabilities = np.clip(predictedProbabilities, epsilon, 1 - epsilon)

    loss = -(1 / numberOfExamples) * np.sum(
        actualYOneHot * np.log(predictedProbabilities)
    )

    return loss


def evaluate(y_true, y_pred, label=""):
    accuracy = np.mean(y_true == y_pred)

    print(f"\n{label}")
    print(f"  Accuracy: {accuracy:.4f}")


X_train, X_test, y_train, y_test = train_test_split(X, actualY, test_size=0.2)

print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size:  {X_test.shape[0]} samples")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class SoftmaxRegression:
    def __init__(self, learning_rate, epochs) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, actualY):
        numberOfExamples = X.shape[0]
        numberOfFeatures = X.shape[1]
        numberOfClasses = len(np.unique(actualY))

        actualYOneHot = one_hot_encode(actualY, numberOfClasses)

        self.weights = np.random.randn(numberOfFeatures, numberOfClasses) * 0.01
        self.bias = np.zeros(numberOfClasses)

        print(f"{'Epoch':<10} {'Loss':<15}")
        print("--" * 20)

        for epoch in range(self.epochs + 1):
            scores = X @ self.weights + self.bias

            predictedProbabilities = softmax(scores)

            loss = cost_function(
                numberOfExamples, actualYOneHot, predictedProbabilities
            )

            self.loss_history.append(loss)

            error = predictedProbabilities - actualYOneHot

            dw = (1 / numberOfExamples) * (X.T @ error)
            db = (1 / numberOfExamples) * np.sum(error, axis=0)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            if epoch % 100 == 0:
                print(f"{epoch:<10} {loss:<15.6f}")

    def predict_proba(self, X):
        scores = X @ self.weights + self.bias
        probabilities = softmax(scores)

        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)

        return np.argmax(probabilities, axis=1)


model = SoftmaxRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)


evaluate(y_train, model.predict(X_train), label="Training Set")
print("--" * 25)
evaluate(y_test, model.predict(X_test), label="Test Set")

print("--" * 25)

print("Learned weights:")
print(model.weights)

print("\nLearned bias:")
print(model.bias)

print("--" * 25)


sk_model = SklearnLogisticRegression(solver="lbfgs", max_iter=1000)

sk_model.fit(X_train, y_train)

print(f"{'Model':<20} {'Train Accuracy':<20} {'Test Accuracy':<20}")
print("--" * 35)

our_train_acc = np.mean(model.predict(X_train) == y_train)
our_test_acc = np.mean(model.predict(X_test) == y_test)

sk_train_acc = sk_model.score(X_train, y_train)
sk_test_acc = sk_model.score(X_test, y_test)

print(f"{'Ours':<20} {our_train_acc:<20.6f} {our_test_acc:<20.6f}")
print(f"{'Sklearn':<20} {sk_train_acc:<20.6f} {sk_test_acc:<20.6f}")
