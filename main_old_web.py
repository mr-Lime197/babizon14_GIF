import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel,  BertTokenizer, BertModel
from pydantic import BaseModel
import torch
import json
from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
model = AutoModel.from_pretrained("squeezebert/squeezebert-uncased")

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained('bert-base-uncased')
with open("static/babiz/link.json", 'r', encoding="UTF-8") as s:
    video=json.load(s)


def get_emb(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return list(embeddings)



def dist(point_1,point_2):
    diff = np.array(point_1) - np.array(point_2)
    return(np.dot(diff, diff))


emb_seq=dict()
for seq in video.keys():
    emb=get_emb(seq)
    emb_seq[seq]=emb

def sim(text):
    emb=get_emb(text)
    best=list(emb_seq.keys())[0]
    for seq in emb_seq.keys():
        if(dist(emb, emb_seq[best])> dist(emb, emb_seq[seq])):
            best=seq
    return best

class Item(BaseModel):
    text: str


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/submit")
async def handle_form(request: Request, text_input: str = Form(...)):
    global video
    s=sim(text_input)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": (text_input, s),
        "video": video[s]
    })
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.89.0.240", port=8080)        
