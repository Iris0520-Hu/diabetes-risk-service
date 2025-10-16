# ml/train_v02.py
import json
import pathlib
import joblib
from math import sqrt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

SEED = 42

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def train(output_dir="artifacts", version="v0.2"):
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # v0.2: 随机森林（对非线性更友好，不需要标准化）
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    ypred = model.predict(Xte)
    score = rmse(yte, ypred)

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.pkl")
    (out / "metrics.json").write_text(
        json.dumps({"version": version, "rmse": float(score)}, indent=2)
    )
    print(f"RMSE={score:.4f}")

if __name__ == "__main__":
    train()
