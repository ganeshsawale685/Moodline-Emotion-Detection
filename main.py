from contextlib import asynccontextmanager
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi import FastAPI,Request
import numpy as np
import re # re-> regex

from keras.src.utils import pad_sequences
from pydantic import BaseModel, Field
from keras.models import load_model
import pickle

from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles



""""
1.We are going to make some constants like:
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequence Length Path
D. Emotion Labels
E. Emotions emojis
"""


model_path = "Artifacts/BiGRU_Model.keras"

tokenizer_path = "Artifacts/tokenizer.pkl"

max_seq_length = 50 # Padding

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


"""
2. Preprocess the upcoming text
Cleans raw texts so it matches the format used while training
A. Convert the text to lowercase
B. Remove punctuation and apostrophes
C. Remove special characters 
D. Remove extra whitespace
E. Remove stopwords
"""

def preprocess(text: str)->str:
    text = text.lower()
    text = re.sub(r"'","",text)
    text = re.sub(r"[^a-z0-9\s]"," ", text)
    text = re.sub(r"\s+", " ",text).strip()
    return text

"""
3. Request and Response Schemas
A. Text Input
B. Prediction Response 
C. Health Response(Server Health Check )

"""

class TextInput(BaseModel):
    text: str = Field(...,min_length=1,max_length=2000,description="Text to be processed",json_schema_extra={"example":"I feel so happy" })

class PredictionResponse(BaseModel):
    text:str
    predicted_emotion:str
    confidence:float
    all_probabilities:dict[str,float]


class HealthResponse(BaseModel):
    status:str
    model_loaded:bool


"""
4. Model Loading and LifeSpan Management
Load the model and toknizer onve the server starts up.
"""

dl_model={}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer")
    dl_model["BiGRU"]=load_model(model_path) # BiGRU Model
    with open("Artifacts/tokenizer.pkl", "rb") as f:
        dl_model['Tokenizer'] = pickle.load(f)   # Tokenizer

    print("Model is Loaded successfully")

    yield  # Pause, model is loaded and server is running and at the point model wait for the request

    dl_model.clear()


'''
5. Mount the static files to the FastAPI application

A.CORS 



'''
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory="static"), name="static")


'''
6. API Endpoints:-
A. Server UI at homepage('/')
B. Health Check Endpoint
C. Predict Emotion Endpoint('/predict')
'''

@app.get('/',include_in_schema=False)
async def serve_ui():
    return FileResponse('static/index.html')


#B. health check endpoint
@app.get('/health',response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is started",model_loaded=True)


# Predict Emotion Endpoint ('/predict')
@app.post('/predict',response_model=PredictionResponse)
def predict(request:TextInput):
    print(request)
    """ 1. Cleans the input sentences.
           2. Convert the words into numeric using tokenizer.
           3. Pad the sequences to ensure uniform length.
           4. Run prediction using the BiGRU model.
           5. Return the top emotion and full probability breakdown.
    """

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")


    if BiGRU_model is None or tokenizer_model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again later.")

    cleaned_text = preprocess(request.text)

    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(tokenized_text, maxlen=max_seq_length,padding="post",truncating="post")

    probabilites = BiGRU_model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilites))

    all_probabilites = {
        label: float(prob) for prob, label in zip(probabilites, emotion_labels)

    }

    return PredictionResponse(
        text=request.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilites[top_emotion_index]),
        all_probabilities=all_probabilites
    )