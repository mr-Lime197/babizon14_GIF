import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel,  BertTokenizer, BertModel
from pydantic import BaseModel
import torch
import json
import logging
from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import  create_engine, Column, Integer, String, LargeBinary, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session, relationship
database="sqlite:///data.db"

class Base(DeclarativeBase): pass
class Gif_added(BaseModel):
    text: str
    video: UploadFile
class Gif_info_db(Base):
    __tablename__="Gif_db"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    file = Column(LargeBinary)
    emb=relationship("info", back_populates="Gif_db")
class Gif_emb_db(Base):
    __tablename__="Gif_emb_db"
    id = Column(Integer, primary_key=True, index=True)
    id_gif=Column(Integer, ForeignKey("Gif_info_db.id"))
    description=Column(String)
    info=relationship("emb", back_populates="Gif_emb_db")


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
class db:
    def __init__(self, path):
        db.__path=path
        db.__engine=create_engine(path, echo=True)
        Base.metadata.create_all(bind=db.__engine)
    def add_description(txt_gif: str, txt_desc: str):
        gif_emb=Gif_emb_db(desription=txt_desc)
        gif_emb.info=txt_gif
    def add_gif(self, Gif:Gif_added):
        gif=Gif_info_db(text=Gif.text, video=Gif.video.file.read())
        with Session(autoflush=False, bind=db.__engine) as s:
            if s.query(Gif_info_db).filter(Gif_info_db.file==gif.file).first():
                logging.error(f'GIFs: {gif.text} is exist')
                return
            s.add(gif)

        
            
                


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
@app.post("/add_gif")
async def add_gif(Gif: Gif_added):
    pass
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=80)