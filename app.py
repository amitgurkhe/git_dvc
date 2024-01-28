from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel


app = FastAPI()

# Load the pre-trained logistic model
model_path = "model.joblib"
#model = load(model_path)

with open(model_path, 'rb') as f:
    model = load(f)

class PredictionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get('/')
def home():
    return "Working fine"

@app.post('/predict')
def predict(inputdata:PredictionInput):
    features =[
        inputdata.Time,
        inputdata.V1,
        inputdata.V2,
        inputdata.V3,
        inputdata.V4,
        inputdata.V5,
        inputdata.V6,
        inputdata.V7,
        inputdata.V8,
        inputdata.V9,
        inputdata.V10,
        inputdata.V11,
        inputdata.V12,
        inputdata.V13,
        inputdata.V14,
        inputdata.V15,
        inputdata.V16,
        inputdata.V17,
        inputdata.V18,
        inputdata.V19,
        inputdata.V20,
        inputdata.V21,
        inputdata.V22,
        inputdata.V23,
        inputdata.V24,
        inputdata.V25,
        inputdata.V26,
        inputdata.V27,
        inputdata.V28,
        inputdata.Amount
    ]
    pred = model.predict([features])[0].item()
    return {'prediction': pred}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=5000)