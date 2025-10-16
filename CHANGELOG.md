# CHANGELOG

## v0.2 — RandomForest 改进
- 模型：`RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)`
- 变化：由线性回归升级为非线性集成模型，能捕捉特征交互，通常对 RMSE 更友好
- 指标对比（同一随机种子、同样的 hold-out 切分）：
  | 版本 | 模型 | RMSE |
  |------|------|------|
  | v0.1 | StandardScaler + LinearRegression | 53.8534 |
  | v0.2 | RandomForestRegressor | **<粘贴 v0.2 的 RMSE>** |
- 结论：RMSE **下降/上升** X.X（约 **YY%**），v0.2 将作为新的默认用于风险排序。

## v0.1 — 基线
- 模型：StandardScaler + LinearRegression
- 指标：RMSE = 53.8534
