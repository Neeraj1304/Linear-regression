import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Custom Linear Regression
# -------------------------------
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            mse = mean_squared_error(y, y_pred)
            self.losses.append(mse)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

# -------------------------------
# Synthetic Data
# -------------------------------
print("\n--- Synthetic Data ---")
X = 2 * np.random.rand(100, 1)
y = 2 * X[:, 0] + 0.5 + np.random.randn(100) * 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

custom_model = CustomLinearRegression(learning_rate=0.1, n_iterations=1000)
custom_model.fit(X_train, y_train)
y_pred_custom = custom_model.predict(X_test)
mse_c, mae_c, r2_c = evaluate_model(y_test, y_pred_custom)

print("\nCustom Model:")
print("  Weights:", custom_model.weights)
print("  Bias:", custom_model.bias)
print("  MSE:", mse_c)
print("  MAE:", mae_c)
print("  R2 :", r2_c)

sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
y_pred_sklearn = sk_model.predict(X_test)
mse_s, mae_s, r2_s = evaluate_model(y_test, y_pred_sklearn)

print("\nSklearn Model:")
print("  Weights:", sk_model.coef_)
print("  Bias:", sk_model.intercept_)
print("  MSE:", mse_s)
print("  MAE:", mae_s)
print("  R2 :", r2_s)

# --- Plots for Synthetic Data ---
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color="blue", label="Actual Data", alpha=0.6)
plt.plot(X_test, y_pred_custom, color="red", label="Custom Model")
plt.plot(X_test, y_pred_sklearn, color="green", linestyle="--", label="Sklearn Model")
plt.title("Synthetic Data - Regression Line Comparison")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_custom, color="red", label="Custom Pred")
plt.scatter(y_test, y_pred_sklearn, color="green", marker="x", label="Sklearn Pred")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black", linestyle="--")
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Synthetic Data - Predicted vs Actual")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(custom_model.losses, color="purple")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Synthetic Data - Custom Model Training Loss Curve")
plt.show()

# -------------------------------
# California Housing Dataset
# -------------------------------
print("\n--- California Housing Dataset ---")
california = fetch_california_housing()
X, y = california.data, california.target

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

custom_model = CustomLinearRegression(learning_rate=0.01, n_iterations=1000)
custom_model.fit(X_train, y_train)
y_pred_custom = custom_model.predict(X_test)
mse_c, mae_c, r2_c = evaluate_model(y_test, y_pred_custom)

print("\nCustom Model:")
print("  Weights:", custom_model.weights[:5], "...")  # print first 5 weights
print("  Bias:", custom_model.bias)
print("  MSE:", mse_c)
print("  MAE:", mae_c)
print("  R2 :", r2_c)

sk_model = LinearRegression()
sk_model.fit(X_train, y_train)
y_pred_sklearn = sk_model.predict(X_test)
mse_s, mae_s, r2_s = evaluate_model(y_test, y_pred_sklearn)

print("\nSklearn Model:")
print("  Weights:", sk_model.coef_[:5], "...")  # print first 5 weights
print("  Bias:", sk_model.intercept_)
print("  MSE:", mse_s)
print("  MAE:", mae_s)
print("  R2 :", r2_s)

# --- Plots for California Housing ---
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_custom, color="red", alpha=0.5, label="Custom Pred")
plt.scatter(y_test, y_pred_sklearn, color="green", alpha=0.5, marker="x", label="Sklearn Pred")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black", linestyle="--")
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("California Housing - Predicted vs Actual")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(custom_model.losses, color="orange")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("California Housing - Custom Model Training Loss Curve")
plt.show()
