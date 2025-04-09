import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Define your custom Echo State Network (ESN)
class SimpleESN:
    def __init__(self, n_input, n_output, n_reservoir=500, spectral_radius=1.25, sparsity=0.1, random_state=None):
        self.n_input = n_input
        self.n_output = n_output
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.random_state = np.random.RandomState(random_state)

        # Input weights
        self.W_in = self.random_state.uniform(-0.5, 0.5, (self.n_reservoir, self.n_input))

        # Reservoir weights
        W = self.random_state.uniform(-0.5, 0.5, (self.n_reservoir, self.n_reservoir))
        mask = self.random_state.rand(*W.shape) < self.sparsity
        W[mask] = 0

        # Scale weights to match spectral radius
        eigenvalues = np.linalg.eigvals(W)
        e_max = np.max(np.abs(eigenvalues))
        self.W = W * (self.spectral_radius / e_max)

        # Output weights
        self.W_out = None
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

        # Discard initial washout
        X = self.states[washout:]
        Y_target = outputs[washout:]

        # Solve W_out using linear regression
        self.W_out = np.linalg.pinv(X) @ Y_target

    def predict(self, inputs):
        n_samples = inputs.shape[0]
        predictions = np.zeros((n_samples, self.n_output))
        state = np.zeros(self.n_reservoir)

        for t in range(n_samples):
            state = self._update(state, inputs[t])
            predictions[t] = np.dot(self.W_out.T, state)

        return predictions

# ------------------------
# Generate a simple dataset
# Let's do binary classification: 0 or 1 depending on sine wave

X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = (np.sin(X).flatten() > 0).astype(int).reshape(-1, 1)  # 1 if sin(x) > 0 else 0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Initialize and train ESN
esn = SimpleESN(
    n_input=1,
    n_output=1,
    n_reservoir=200,
    spectral_radius=1.2,
    sparsity=0.1,
    random_state=42
)

print("Training the ESN...")
esn.fit(X_train, y_train)

# ------------------------
# Prediction
print("Predicting...")
y_pred = esn.predict(X_test)

# ------------------------
# Threshold predictions to 0 or 1
y_pred_class = (y_pred.flatten() > 0.5).astype(int)

# ------------------------
# Results
print("\n==== Weights ====")
print("Input Weights (W_in):\n", esn.W_in)
print("\nReservoir Weights (W):\n", esn.W)
print("\nOutput Weights (W_out):\n", esn.W_out)

print("\n==== Predictions ====")
print("Predicted Values (before threshold):\n", y_pred.flatten())
print("\nPredicted Classes (after threshold):\n", y_pred_class)
print("\nTrue Labels:\n", y_test.flatten())

# ------------------------
# Accuracy and MSE
acc = accuracy_score(y_test.flatten(), y_pred_class)
mse = mean_squared_error(y_test.flatten(), y_pred.flatten())

print(f"\nAccuracy: {acc * 100:.2f}%")
print(f"Mean Squared Error: {mse:.4f}")
