import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR

np.random.seed(42)


def generateSynData(rows):
    # We deliberately use a nonlinear relationship (sine curve) as the ground
    X = 2 * np.pi * np.random.rand(rows, 1)
    noise = 0.3 * np.random.randn(rows, 1)
    y = np.sin(X) + noise
    return X, y


X, y = generateSynData(200)


def train_test_split(X, y, test_size=0.2):
    # Shuffle indices so the split is random and not ordered by X value.
    # Then carve off the last (test_size * 100)% as the test set.
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    split = int(m * (1 - test_size))

    trainIdx = indices[:split]
    testIdx = indices[split:]

    return X[trainIdx], X[testIdx], y[trainIdx], y[testIdx]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size:  {X_test.shape[0]} samples")
print("--" * 25)


def evaluate(y_true, y_pred, label=""):
    # Flatten both arrays so shapes never cause broadcasting issues.
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    errors = y_pred - y_true

    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    # R² = 1 means perfect predictions. R² = 0 means the model is no better
    r2 = 1 - (np.sum(errors**2) / np.sum((y_true - np.mean(y_true)) ** 2))

    print(f"\n{label}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")


def gaussian_kernel(X_train, x_query, tau):

    distance_squared = (X_train - x_query) ** 2
    exponent = distance_squared / (2 * tau**2)
    weights = np.exp(-exponent)  # shape (m, 1)
    return weights.flatten()  # shape (m,)


def build_design_matrix(X):
    # combining the features with the bias ( value = 1)
    # First coloum is the bias ( all values are 1 )
    m = X.shape[0]
    ones = np.ones((m, 1))
    return np.hstack([ones, X])  # shape (m, 2)


def build_weight_matrix(weights):
    return np.diag(weights)  # shape (m, m)


class LocallyWeightedRegression:
    def __init__(self, tau):
        self.tau = tau
        self.X_train = None
        self.y_train = None
        self.featureMatrix = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.featureMatrix = build_design_matrix(X)  # built once, reused every query

        print(f"{'Locally Weighted Regression':^50}")
        print("--" * 25)
        print(f"  Training examples stored : {X.shape[0]}")
        print(f"  Bandwidth (τ)            : {self.tau}")
        print("Note: no weight learned — model solves per query at predict time")
        print("--" * 25)

    def _predict_single(self, x_query):
        weights = gaussian_kernel(self.X_train, x_query, self.tau)  # (m,)

        weightMatrix = build_weight_matrix(weights)  # (m, m)

        XtW = self.featureMatrix.T @ weightMatrix  # (2, m)
        XtWX = XtW @ self.featureMatrix  # (2, 2) — small, cheap to solve
        XtWy = XtW @ self.y_train  # (2, 1)
        theta = np.linalg.solve(XtWX, XtWy)  # (2, 1)

        x_q_design = np.array([[1.0, x_query.flatten()[0]]])  # (1, 2)
        return (x_q_design @ theta)[0][0]

    def predict(self, X_query):
        m = X_query.shape[0]
        predictions = np.zeros(m)

        for i in range(m):
            predictions[i] = self._predict_single(X_query[i].reshape(1, -1))

            if (i + 1) % 10 == 0 or (i + 1) == m:
                print(f"  Predicting... {i + 1}/{m} done", end="\r")

        print()
        return predictions.reshape(-1, 1)


# FIT AND PREDICT

model = LocallyWeightedRegression(tau=0.5)
model.fit(X_train, y_train)

print("\nRunning predictions on training set...")
y_train_pred = model.predict(X_train)

print("Running predictions on test set...")
y_test_pred = model.predict(X_test)


print("\n" + "=" * 50)
print("LOCALLY WEIGHTED REGRESSION")
evaluate(y_train, y_train_pred, label="Training Set")
print("--" * 25)
evaluate(y_test, y_test_pred, label="Test Set")
print("\n" + "=" * 50)


sk_model = SklearnLR()
sk_model.fit(X_train, y_train)

sk_train_pred = sk_model.predict(X_train)
sk_test_pred = sk_model.predict(X_test)

print("SKLEARN GLOBAL LINEAR REGRESSION (baseline)")
evaluate(y_train, sk_train_pred, label="Training Set")
print("--" * 25)
evaluate(y_test, sk_test_pred, label="Test Set")


def r2_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    errors = y_pred - y_true
    return 1 - (np.sum(errors**2) / np.sum((y_true - np.mean(y_true)) ** 2))


lwr_train_r2 = r2_score(y_train, y_train_pred)
lwr_test_r2 = r2_score(y_test, y_test_pred)
sk_train_r2 = r2_score(y_train, sk_train_pred)
sk_test_r2 = r2_score(y_test, sk_test_pred)

print("\n" + "=" * 50)
print(f"{'FINAL COMPARISON':^50}")
print("=" * 50)
print(f"\n{'Model':<30} {'Train R²':<15} {'Test R²':<15}")
print("--" * 25)
print(f"{'LWR (τ = 0.5)':<30} {lwr_train_r2:<15.4f} {lwr_test_r2:<15.4f}")
print(f"{'Sklearn Linear Regression':<30} {sk_train_r2:<15.4f} {sk_test_r2:<15.4f}")


# TAU SENSITIVITY STUDY
# As tau increases, LWR smoothly degrades toward global OLS because distant

print("\n" + "=" * 50)
print(f"{'TAU SENSITIVITY (Test R²)':<50}")
print("=" * 50)
print(f"\n{'τ value':<15} {'Test R²':<15} {'Character':<25}")
print("--" * 25)

tau_notes = {
    0.1: "very local — wiggly, prone to overfit",
    0.3: "local — follows curve closely",
    0.5: "balanced — our default",
    1.0: "wide — smoothing out features",
    2.0: "very wide — approaching global line",
    5.0: "global — nearly identical to OLS",
}

for tau_val, note in tau_notes.items():
    tau_model = LocallyWeightedRegression(tau=tau_val)
    tau_model.fit(X_train, y_train)
    tau_preds = tau_model.predict(X_test)
    tau_r2 = r2_score(y_test, tau_preds)
    print(f"{tau_val:<15} {tau_r2:<15.4f} {note:<25}")


# MODEL SUMMARY

print("\n" + "=" * 50)
print(f"{'MODEL SUMMARY':^50}")
print("=" * 50)
print(f"""
  Algorithm         : Locally Weighted Regression
  Kernel            : Gaussian
  Bandwidth (τ)     : 0.5
  Parameters stored : None (lazy learner)
  Training cost     : O(1) — just memorise data
  Prediction cost   : O(m) per query point
                      (one full normal equation solve each time)
  Data              : y = sin(x) + noise
  Training samples  : {X_train.shape[0]}
  Test samples      : {X_test.shape[0]}
""")
