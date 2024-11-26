from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import HousePricePredictor

app = FastAPI()
predictor = HousePricePredictor()

class HouseFeatures(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    GarageArea: int
    TotalBsmtSF: int
    FullBath: int
    YearBuilt: int

@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        prediction = predictor.predict(features.dict())
        return {"predicted_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
