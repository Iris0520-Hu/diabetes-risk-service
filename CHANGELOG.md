# CHANGELOG

## v0.2.1 — Improved model with SVR(RBF)
- Switched model from LinearRegression to SVR (RBF kernel) with StandardScaler
- Added small grid search to optimize C, gamma, epsilon
- Improved model generalization without overfitting

| Version | Model | RMSE |
|----------|-------------------------------|---------|
| v0.1     | StandardScaler + LinearRegression | 53.8534 |
| v0.2.1   | StandardScaler + SVR(RBF) + GridSearch | 52.4928 |

RMSE improved by 1.36 (2.52% lower than v0.1)

---

## v0.1 — Baseline model
- First version with linear regression
- Metric: RMSE = 53.8534
