# Reservoir Computing Project

## 🧠 Overview

This project implements **Reservoir Computing (RC)**, a computational framework particularly effective for solving complex time-series tasks. RC leverages the dynamic properties of recurrent systems and is a lightweight alternative to traditional deep learning models like LSTMs or GRUs.

---

## ✅ Work Done So Far

- **Data Preprocessing**:
  - Loaded historical temperature data (CSV file) ✅
  - Handled missing values, smoothing, and normalized the data ✅
  - Visualized time series trends ✅

- **Initial Reservoir Setup**:
  - Built a random, fixed recurrent neural network (the *reservoir*) ✅
  - Initialized input and reservoir weight matrices ✅
  - Defined activation function (e.g., `tanh`) for internal dynamics ✅

- **Simple Readout Layer**:
  - Linear regression layer to map reservoir states to output ✅

- **Testing Basic Flow**:
  - Passed the data through the reservoir
  - Collected internal states
  - Trained a linear readout on reservoir outputs ✅

---

## 🚀 What’s Next

- **Hyperparameter Tuning**:
  - Optimize reservoir size, spectral radius, sparsity
- **Noise Regularization**:
  - Add controlled noise during reservoir updates
- **Evaluation**:
  - Implement RMSE, MAE, and MSE metrics for prediction quality
- **Advanced Readout**:
  - Explore logistic regression / ridge regression as readout alternatives
- **Save Models**:
  - Save trained reservoirs and readouts for inference
- **Deployment (optional)**:
  - Build a lightweight API to serve predictions

---

## 🏗️ Planned Neural Network Architecture
