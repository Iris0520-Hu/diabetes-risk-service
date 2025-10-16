import pathlib
import joblib
from fastapi import FastAPI, HTTPException
from app.models import PredictRequest, PredictResponse
import os

ART_DIR = pathlib.Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ART_DIR / "model.pkl"
# MODEL_VERSION = "v0.1"  # v0.2 时再改
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

app = FastAPI(title="Virtual Diabetes Clinic Risk Service", version=MODEL_VERSION)
_model = None  # 简单的进程内缓存

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("artifacts/model.pkl not found. Run training first.")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    exists = MODEL_PATH.exists()
    return {"status": "ok" if exists else "missing_model", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = load_model()
        x = [[req.age, req.sex, req.bmi, req.bp, req.s1, req.s2, req.s3, req.s4, req.s5, req.s6]]
        yhat = float(model.predict(x)[0])
        return PredictResponse(prediction=yhat)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "bad_input", "message": str(e)})
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail={"error": "model_missing", "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "internal", "message": str(e)})
