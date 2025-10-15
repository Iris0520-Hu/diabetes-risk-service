import pathlib, json, joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

SEED = 42
def rmse(y_true, y_pred): return sqrt(mean_squared_error(y_true, y_pred))

def train(output_dir="artifacts", version="v0.1"):
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xte)
    score = rmse(yte, ypred)

    out = pathlib.Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.pkl")
    (out / "metrics.json").write_text(json.dumps({"version": version, "rmse": float(score)}, indent=2))
    print(f"RMSE={score:.4f}")

if __name__ == "__main__":
    train()
