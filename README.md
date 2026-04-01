# Linear Regression From Scratch (Gradient Descent)

## Overview

Implemented Linear Regression from scratch using NumPy to understand how models learn through optimization.
Built the full pipeline including preprocessing, feature scaling, gradient descent, and evaluation.

---

## Problem

Predict housing prices using structured numerical features.

---

## Approach

### Model

* Implemented Linear Regression manually
* Used Gradient Descent for optimization
* Initialized weights and bias
* Iteratively minimized loss

### Pipeline

* Data loading and cleaning
* Feature scaling (standardization)
* Train-test split
* Model training
* Evaluation

---

## Core Concepts Implemented

* Linear Model:
  `y = w·x + b`

* Loss Function:
  Mean Squared Error (MSE)

* Optimization:
  Gradient Descent

* Feature Scaling:
  Standardization for stable training

---

## Results

* RMSE: ~50,000
* MAE: ~40,000
* R² Score: ~0.57

---

## Validation

Compared results with sklearn’s Linear Regression:

* Achieved similar performance
* Verified correctness of implementation

---

## Key Learnings

* Gradient descent drives model learning through iterative updates
* Feature scaling is essential for convergence
* Model performance depends more on features than algorithm choice
* Perfect metrics often indicate bugs or data leakage

---

## Limitations

* Linear model cannot capture complex non-linear relationships
* Performance limited by feature quality
* Sensitive to outliers

---

## How to Run

```bash
python test.py
```

---

## Project Structure

```text
linreg-system/
├── preprocessing.py
├── model.py
├── test.py
```

---

## Future Improvements

* Add regularization (Ridge/Lasso)
* Try polynomial features
* Use tree-based models
* Improve feature engineering

---

## Summary

This project focuses on understanding **how machine learning models work internally**, rather than just using libraries.
It demonstrates the full lifecycle of building and validating a model from scratch.
