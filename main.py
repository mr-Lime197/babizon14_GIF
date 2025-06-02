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
class FileMeta(BaseModel):
    text: str
class Gif_added(BaseModel):
    text: str
    video: bytes
class Gif_emb_db(Base):
    __tablename__="Gif_emb_db"
    id = Column(Integer, primary_key=True, index=True)
    id_gif=Column(Integer, ForeignKey("Gif_info_db.id"))
    description=Column(String)
    embeding=Column(LargeBinary)
    info=relationship("Gif_info_db", back_populates="emb")
class Gif_info_db(Base):
    __tablename__="Gif_info_db"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    file = Column(LargeBinary)
    emb=relationship("Gif_emb_db", back_populates="info")

class Model:
    def __init__(self):
        self.__tokenizer=AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
        self.__model=AutoModel.from_pretrained("squeezebert/squeezebert-uncased")
    def get_emb(self, text: str) -> bytes:
        inputs = self.__tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.__model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)\
            .squeeze()\
            .numpy()
        return embeddings.tobytes()
    def dist(point_1,point_2):
        diff = np.array(point_1) - np.array(point_2)
        return(np.dot(diff, diff))
    

#tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")
#model = AutoModel.from_pretrained("squeezebert/squeezebert-uncased")

# emb_seq=dict()
# for seq in video.keys():
#     emb=get_emb(seq)
#     emb_seq[seq]=emb

# def sim(text):
#     emb=get_emb(text)
#     best=list(emb_seq.keys())[0]
#     for seq in emb_seq.keys():
#         if(dist(emb, emb_seq[best])> dist(emb, emb_seq[seq])):
#             best=seq
#     return best
class db:
    def __init__(self, path:str, model:Model):
        db.__path=path
        db.__model=model
        db.__engine=create_engine(path)
        Base.metadata.create_all(bind=db.__engine)
    def add_description(self, txt_desc: str, gif_id: int):
        with Session(autoflush=False, bind=db.__engine) as s:
            gif_emb=Gif_emb_db(description=txt_desc, embeding=db.__model.get_emb(txt_desc))
            if s.query(Gif_info_db).filter(Gif_info_db.id==gif_id).count()==0:
                logging.error(f"gif with id {gif_id} is not exist")
                return
            gif_info=s.query(Gif_info_db).filter(Gif_info_db.id==gif_id).first()
            gif_emb.info=gif_info
            s.add(gif_emb)
            logging.info(f"description {txt_desc} add for gif {gif_id}")
            s.commit()
        return
    def add_gif(self, Gif:Gif_added):
        gif=Gif_info_db(text=Gif.text, file=Gif.video)
        with Session(autoflush=False, bind=db.__engine) as s:
            if s.query(Gif_info_db).filter(Gif_info_db.file==gif.file).first():
                logging.error(f'GIFs: {gif.text} is exist')
                return
            s.add(gif)
            logging.info(f"added gif {Gif.text}")
            s.commit()
            self.add_description(txt_desc=gif.text, gif_id=gif.id)
        return
    def get_id_gif(self, file:File):
        with Session(autoflush=False, bind=db.__engine) as s:
            if s.query(Gif_info_db).filter(Gif_info_db.file==file).count()==0:
                logging.error("gif is not exist")
                return
            return s.query(Gif_info_db).filter(Gif_info_db.file==file).first().id
            
            

        

        
            
                


# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
# @app.get("/")
# async def read_form(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
# @app.post("/submit")
# async def handle_form(request: Request, text_input: str = Form(...)):
#     global video
#     s=sim(text_input)
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "result": (text_input, s),
#         "video": video[s]
#     })
# @app.post("/add_gif")
# async def add_gif(Gif: Gif_added):
#     pass
app = FastAPI()
model=Model()
base=db(database, model)
@app.post("/file/upload-file")
def upload_file_bytes(
    file: UploadFile = File(...),
    meta: str = Form(...),
):
    meta_data = FileMeta(**json.loads(meta))
    base.add_gif(Gif_added(text=meta_data.text, video=file.file.read()))
    return
@app.post("/file/add_desc")
def add_desc(gif: Gif_added):
    id=base.get_id_gif(bytes(gif.video))
    base.add_description(gif.text, id)
    return
# def upload_file(file: UploadFile):
#   base.add_gif(Gif_added(text="gay2", video=file.file.read()))
#  return 
if __name__ == "__main__":
     logging.basicConfig(filename='lg.log')
     import uvicorn
     uvicorn.run(app, host="127.0.0.1", port=80)