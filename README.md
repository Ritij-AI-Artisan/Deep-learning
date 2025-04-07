# Reservoir Computing Implementation

This project implements a basic **Reservoir Computing** model.

Reservoir Computing is a type of **recurrent neural network (RNN)** where only the output layer is trained. It is especially effective for time-series prediction tasks because the reservoir (a fixed, randomly connected network) nonlinearly transforms inputs into a high-dimensional space.

---

## What is Reservoir Computing?

**Reservoir Computing** uses the following core components:

- **Input Layer:** Maps the input into the reservoir.
- **Reservoir (Dynamic System):** A fixed, high-dimensional, randomly connected recurrent network. No training happens here.
- **Output Layer (Readout):** A simple linear or ridge regression that is trained based on reservoir states to produce predictions.

The reservoir itself transforms simple input signals into a rich set of dynamics that make the final prediction task easier.

---

## Mathematical Formulation

1. **Reservoir Dynamics**:

\[
x(t+1) = (1 - \alpha) x(t) + \alpha \tanh(W_{\text{in}} u(t) + W x(t))
\]

where:
- \(x(t)\) = internal reservoir state at time \(t\)
- \(u(t)\) = input signal at time \(t\)
- \(W_{\text{in}}\) = input-to-reservoir weight matrix
- \(W\) = recurrent reservoir weight matrix
- \(\alpha\) = leaking rate (controls speed of reservoir)

2. **Training Output Weights**:

Once the reservoir states are collected, a simple Ridge Regression (or Linear Regression) is applied to map reservoir states to output targets.

---

## Current Progress

- [x] Load a time-series dataset (e.g., temperature, stock prices)
- [x] Preprocess the data (normalization using MinMaxScaler)
- [x] Build reservoir computing model:
  - Randomly initialized input weights
  - Randomly initialized reservoir (sparse connections)
  - Reservoir update dynamics
- [x] Train output weights using **Ridge Regression**
- [x] Evaluate on unseen test data (Mean Squared Error)
- [x] Visualize predicted vs actual

---

## To Be Done

- [ ] Implement Hyperparameter Tuning (grid search over spectral radius, leaking rate, etc.)
- [ ] Add support for multi-step ahead forecasting
- [ ] Add support for saving and loading trained models
- [ ] Add evaluation metrics (e.g., R² score, MAE)
- [ ] Add documentation and packaging

---

## Reservoir Computing Model Architecture

```text
Input Data
   │
   ▼
Input Weights (Win)
   │
   ▼
Reservoir (Random Recurrent Network, fixed weights W)
   │
   ▼
Output Weights (trained via Ridge Regression)
   │
   ▼
Predicted Output
```

---

## Python Script Overview

Everything is inside a **single Python file** (`reservoir_computing.py`).

Main Components:

- **Load and Preprocess Data:** Load CSV file, scale data between -1 and 1.
- **Initialize Reservoir:** Random initialization of input and internal weights.
- **Train:** Collect reservoir states and train output weights with Ridge Regression.
- **Test:** Use trained weights to predict on new inputs.
- **Evaluate:** Plot actual vs predicted, calculate Mean Squared Error.

---

## Full Python Script

```python
# reservoir_computing.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1. Load and Preprocess Data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data = data.iloc[:, 0].values.reshape(-1, 1)  # Assume single-column time series
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# 2. Reservoir Computing Class
class ReservoirComputing:
    def __init__(self, input_size, reservoir_size=500, spectral_radius=0.9, sparsity=0.1, alpha=0.3, ridge_lambda=1e-6):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.alpha = alpha
        self.ridge_lambda = ridge_lambda
        self._initialize_weights()

    def _initialize_weights(self):
        self.Win = np.random.uniform(-1, 1, (self.reservoir_size, self.input_size))
        W = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        W[np.random.rand(*W.shape) > self.sparsity] = 0  # sparsity
        spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (self.spectral_radius / spectral_radius)

    def _update(self, state, input_signal):
        pre_activation = np.dot(self.Win, input_signal) + np.dot(self.W, state)
        return (1 - self.alpha) * state + self.alpha * np.tanh(pre_activation)

    def fit(self, inputs, targets, washout=50):
        states = np.zeros((inputs.shape[0], self.reservoir_size))
        state = np.zeros((self.reservoir_size, 1))

        for t in range(inputs.shape[0]):
            state = self._update(state, inputs[t])
            states[t] = state[:, 0]

        self.model = Ridge(alpha=self.ridge_lambda, fit_intercept=False)
        self.model.fit(states[washout:], targets[washout:])

    def predict(self, inputs, washout=50):
        states = np.zeros((inputs.shape[0], self.reservoir_size))
        state = np.zeros((self.reservoir_size, 1))

        for t in range(inputs.shape[0]):
            state = self._update(state, inputs[t])
            states[t] = state[:, 0]

        predictions = self.model.predict(states[washout:])
        return predictions

# 3. Main Execution
if __name__ == "__main__":
    # Load and preprocess
    filepath = "path_to_your_dataset.csv"  # Update this path
    data = load_data(filepath)
    scaled_data, scaler = preprocess_data(data)

    # Split into train/test
    train_size = int(0.8 * len(scaled_data))
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Prepare inputs and targets
    X_train = train_data[:-1]
    y_train = train_data[1:]
    X_test = test_data[:-1]
    y_test = test_data[1:]

    # Initialize and Train
    rc = ReservoirComputing(input_size=1)
    rc.fit(X_train, y_train)

    # Predict
    y_pred = rc.predict(X_test)

    # Inverse scale to original
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Evaluation
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print(f"Test MSE: {mse:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="Predicted")
    plt.title("Reservoir Computing Prediction")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
```

---

## How to Run

1. Place your dataset CSV file in the project directory.
2. Update the `filepath = "path_to_your_dataset.csv"` in the script.
3. Run:

```bash
python reservoir_computing.py
```

---

## References

- Jaeger, H. (2001). "The Echo State Approach to Analysing and Training Recurrent Neural Networks"
- Lukosevicius, M., & Jaeger, H. (2009). "Reservoir computing approaches to recurrent neural network training"
