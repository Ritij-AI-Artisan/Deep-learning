import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge

class SimpleESN:
    def __init__(self, n_input, n_output, n_reservoir=500, spectral_radius=1.25, sparsity=0.1, alpha=1.0, random_state=None):
        self.n_input = n_input
        self.n_output = n_output
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.alpha = alpha
        self.random_state = np.random.RandomState(random_state)

        
        self.W_in = self.random_state.uniform(-0.5, 0.5, (self.n_reservoir, self.n_input))

        
        W = self.random_state.uniform(-0.5, 0.5, (self.n_reservoir, self.n_reservoir))
        mask = self.random_state.rand(*W.shape) < self.sparsity
        W[mask] = 0

        eigenvalues = np.linalg.eigvals(W)
        e_max = np.max(np.abs(eigenvalues))
        self.W = W * (self.spectral_radius / e_max)

        self.ridge_model = Ridge(alpha=self.alpha)
        self.states = None

    def _update(self, state, input_signal):
        preactivation = np.dot(self.W_in, input_signal) + np.dot(self.W, state)
        return np.tanh(preactivation)

    def fit(self, inputs, outputs, washout=50):
        n_samples = inputs.shape[0]
        self.states = np.zeros((n_samples, self.n_reservoir))
        state = np.zeros(self.n_reservoir)

        for t in range(n_samples):
            state = self._update(state, inputs[t])
            self.states[t] = state

        X = self.states[washout:]
        Y = outputs[washout:]

        self.ridge_model.fit(X, Y)

    def predict(self, inputs):
        n_samples = inputs.shape[0]
        predictions = np.zeros((n_samples, self.n_output))
        state = np.zeros(self.n_reservoir)

        for t in range(n_samples):
            state = self._update(state, inputs[t])
            predictions[t] = self.ridge_model.predict(state.reshape(1, -1))

        return predictions



X = np.linspace(0, 10, 10000).reshape(-1, 1)
y = (np.sin(X).flatten() > 0).astype(int).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


esn = SimpleESN(
    n_input=1,
    n_output=1,
    n_reservoir=500,
    spectral_radius=1.2,
    sparsity=0.1,
    alpha=0.5,
    random_state=42
)

n_epochs = 10
train_losses = []
test_losses = []

for epoch in range(n_epochs):
    esn.fit(X_train, y_train, washout=100)

    y_train_pred = esn.predict(X_train)
    y_test_pred = esn.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_losses.append(train_mse)
    test_losses.append(test_mse)

    print(f"Epoch {epoch+1}/{n_epochs} | Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f}")


y_pred = esn.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)


acc = accuracy_score(y_test, y_pred_class)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")


plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(X_test.flatten(), y_test.flatten(), '.', label="True", alpha=0.5)
plt.plot(X_test.flatten(), y_pred.flatten(), '.', label="Predicted", alpha=0.5)
plt.title("Real vs Predicted (Dense Plot)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs+1), train_losses, label="Train MSE")
plt.plot(range(1, n_epochs+1), test_losses, label="Test MSE")
plt.title("Training & Testing Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

