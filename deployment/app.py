from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import torch
import numpy as np
import pandas as pd
from separate_data_test import preprocess, new_features
from ModelLoader import ModelLoader  # Assuming you have your model loader code in ModelLoader.py

#SASA_MAX = 6
#ERDEM_MAX = 3
LABELS = {
    0: 'irrelevant',
    1: 'SASA',
    2: 'Erdemoğlu'
}

# Initialize FastAPI
app = FastAPI()

# Define request body model
class TextsRequest(BaseModel):
    texts: List[str]

# Prediction route
@app.post("/predict")
def predict(request: TextsRequest):
    texts = request.texts
    model_loader = ModelLoader()
    embedder = model_loader.embedder
    model = model_loader.model
    
    # Preprocessing texts
    texts_df = pd.DataFrame(texts, columns=['tweet_text'])
    texts_df = new_features(texts_df)
    preprocessed_df = preprocess(texts_df)
    
    features = preprocessed_df.iloc[:,1:]
    features = features.to_numpy()
    
    preprocessed_texts = preprocessed_df['tweet_text'].tolist()
    embeds = embedder.encode(preprocessed_texts)
    embeds = np.concatenate((embeds, features), axis=1)
    embeds = torch.Tensor(embeds)
    
    prediction = model(embeds)
    prediction = torch.softmax(prediction, 1)
    scores, predicted = torch.max(prediction, 1)
    predicted = predicted.tolist()
    scores = scores.tolist()
    for i in range(len(predicted)):
        predicted[i] = LABELS.get(predicted[i])
    
    return {"predictions": predicted, "score": scores}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        reload = True
)