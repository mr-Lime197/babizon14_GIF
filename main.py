import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
import torch
import json
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

def get_emb(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return list(embeddings)
def dist(point_1,point_2):
    diff = np.array(point_1) - np.array(point_2)
    return(np.dot(diff, diff))
def sim(text):
    with open("data.json", "r", encoding="UTF-8") as s:
        emb_seq=json.load(s)[0]
    emb=get_emb(text)
    best=list(emb_seq.keys())[0]
    for seq in emb_seq.keys():
        if(dist(emb, emb_seq[best])> dist(emb, emb_seq[seq])):
            best=seq
    return best

class Item(BaseModel):
    text: str

app = FastAPI()
templates = Jinja2Templates(directory="templates")
@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/submit")
async def handle_form(request: Request, text_input: str = Form(...)):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": (text_input, sim(text_input))
    })
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.89.169", port=80)