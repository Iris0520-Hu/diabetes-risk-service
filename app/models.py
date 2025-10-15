from pydantic import BaseModel, field_validator

class PredictRequest(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    # 避免传字符串时报错，统一转 float
    @field_validator("*", mode="before")
    @classmethod
    def to_float(cls, v):
        return float(v)

class PredictResponse(BaseModel):
    prediction: float
