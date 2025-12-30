from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import StringConstraints
from typing import Annotated
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
import os
from mangum import Mangum

app = FastAPI()

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class PredictRequest(BaseModel):
    text: Annotated[str, StringConstraints(min_length=1)]


class PredictResponse(BaseModel):
    prediction: str


class ModelWrapper:
    def __init__(self):
        model_dir = os.path.join(os.getcwd(), "model")

        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=512)
        self.tokenizer.enable_truncation(max_length=512)

        self.embedding_session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx")
        )
        self.classifier_session = ort.InferenceSession(
            os.path.join(model_dir, "classifier.onnx")
        )


model_wrapper = ModelWrapper()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    encoded = model_wrapper.tokenizer.encode(request.text)

    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

    embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    embeddings = model_wrapper.embedding_session.run(None, embedding_inputs)[0]

    classifier_input_name = model_wrapper.classifier_session.get_inputs()[0].name
    classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}

    prediction = model_wrapper.classifier_session.run(None, classifier_inputs)[0]

    label_idx = prediction[0]
    label_str = SENTIMENT_MAP.get(label_idx, "unknown")

    return PredictResponse(prediction=label_str)


handler = Mangum(app)
