from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel 
from test import predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# For handling the request data
class BeerReview(BaseModel):
    abv: float
    beerId: int
    brewerId: int
    appearance: float
    aroma: float
    palate: float
    taste: float
    style: str
    reviewText: str

@app.post("/predict/")
async def get_prediction(review: BeerReview):
    prediction = predict(
        abv=review.abv,
        beerId=review.beerId,
        brewerId=review.brewerId,
        appearance=review.appearance,
        aroma=review.aroma,
        palate=review.palate,
        taste=review.taste,
        style=review.style,
        reviewText=review.reviewText
    )

    # Convert numpy types to python types to return them
    if isinstance(prediction, np.generic):
        prediction = prediction.item()

    return {"predicted_rating": prediction}