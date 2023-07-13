# from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 1 if torch.cuda.is_available() else None
    # model = pipeline('fill-mask', model='bert-base-uncased', device=device)
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    texts = model_inputs.get('text', None)
    if texts == None:
        return {'message': "No text provided"}
    
    # Run the model
    result = model.encode(texts)

    # Return the results as a dictionary
    return result
