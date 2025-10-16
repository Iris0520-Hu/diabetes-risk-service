# ml/train_v02.py
import json
import pathlib
import joblib
from math import sqrt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

SEED = 42

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def train(output_dir="artifacts", version="v0.2"):
    # 1) 数据
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # 2) 模型：StandardScaler + SVR(RBF) + 小网格搜索（速度快且稳定）
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ])

    param_grid = {
        "svr__C": [3, 10, 30],
        "svr__gamma": ["scale", 0.03, 0.1],
        "svr__epsilon": [0.1, 0.2],
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=1,      # SVR 本身不并行，避免 CI 并发问题
        refit=True,
    )
    grid.fit(Xtr, ytr)

    # 3) 评估
    ypred = grid.predict(Xte)
    score = rmse(yte, ypred)

    # 4) 输出工件
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(grid.best_estimator_, out / "model.pkl")
    (out / "metrics.json").write_text(
        json.dumps(
            {"version": version, "rmse": float(score), "best_params": grid.best_params_},
            indent=2
        )
    )
    print(f"Best params: {grid.best_params_}")
    print(f"RMSE={score:.4f}")

if __name__ == "__main__":
    train()
