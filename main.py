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
from sqlalchemy import  create_engine, Column, Integer, String, LargeBinary, ForeignKey, delete
from sqlalchemy.orm import DeclarativeBase, Session, relationship
from fastapi.responses import FileResponse
from fastapi.responses import Response
import io
database="sqlite:///data.db"

class Base(DeclarativeBase): pass
class FileMeta(BaseModel):
    text: str
class Desc(BaseModel):
    lst:list
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
        self.__tokenizer=AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.__model=AutoModel.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    def get_emb(self, text: str) -> bytes:
        inputs = self.__tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.__model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)\
            .squeeze()\
            .numpy()
        return embeddings.tobytes()
    def dist(self, point_1,point_2):
        diff = np.array(point_1) - np.array(point_2)
        return(np.dot(diff, diff))
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
                return {"status":404}
            gif_info=s.query(Gif_info_db).filter(Gif_info_db.id==gif_id).first()
            gif_emb.info=gif_info
            s.add(gif_emb)
            logging.info(f"description {txt_desc} add for gif {gif_id}")
            s.commit()
        return {"status":200}
    def add_gif(self, Gif:Gif_added) -> dict:
        gif=Gif_info_db(text=Gif.text, file=Gif.video)
        with Session(autoflush=False, bind=db.__engine) as s:
            if s.query(Gif_info_db).filter(Gif_info_db.file==gif.file).first():
                logging.error(f'GIFs: {gif.text} is exist')
                return {"status":404}
            s.add(gif)
            logging.info(f"added gif {Gif.text}")
            s.commit()
            self.add_description(txt_desc=gif.text, gif_id=gif.id)
        return {"status":200}
    

    def get_id_gif(self, file:bytes) -> dict:
        with Session(autoflush=False, bind=db.__engine) as s:
            if s.query(Gif_info_db).filter(Gif_info_db.file==file).count()==0:
                logging.error("gif is not exist")
                return {"status":404, "data":None}
            return {"status":200, "data":s.query(Gif_info_db).filter(Gif_info_db.file==file).first().id}
        

    def get_sim(self, text: str) -> dict:
        model=self.__model
        bestsim=0
        bestid=-1
        emb=np.frombuffer(model.get_emb(text), dtype=np.float32)
        with Session(autoflush=False, bind=self.__engine) as db:
            h=db.query(Gif_emb_db)
            if h.count()==0:
                logging.error("database is empty")
                return {"status":500, "data":None}
            for row in h:
                curemb=np.frombuffer(row.embeding, dtype=np.float32)
                sim=model.dist(emb, curemb)
                if(sim<bestsim or bestid==-1):
                    bestsim=sim
                    bestid=row.id_gif
            h=db.query(Gif_info_db).filter(Gif_info_db.id==bestid).first()
            return {"status":200, "data":(h.text, h.file)}
        

    def get_cnt_desc(self, gif_id: int) -> dict:
        with Session(autoflush=False, bind=db.__engine) as s:
            return {"status":200, "data":s.query(Gif_emb_db).filter(Gif_emb_db.id_gif==gif_id).count()}
        

    def del_gif(self, gif_id:int) ->dict:
        cnt=self.get_cnt_desc(gif_id)
        if cnt["status"]==200 and cnt["data"]>0:
            with Session(autoflush=False, bind=db.__engine) as s:
                s.query(Gif_info_db).filter(Gif_info_db.id==gif_id).delete()
                s.query(Gif_emb_db).filter(Gif_emb_db.id_gif==gif_id).delete()
                s.commit()
            return {"status":200}
        return {"status":404}
app = FastAPI()
model=Model()
base=db(database, model)
@app.post("/file/upload-file")
def upload_file_bytes(
    file: UploadFile = File(...),
    meta: str = Form(...),
):
    meta_data = FileMeta(**json.loads(meta))
    status = base.add_gif(Gif_added(text=meta_data.text, video=file.file.read()))
    return Response(status_code=status["status"])
@app.post("/file/add_desc")
def add_desc(
    file: UploadFile = File(...),
    meta: str = Form(...),
):
    meta_data = Desc(**json.loads(meta))
    dt=base.get_id_gif(file.file.read())
    if(dt["status"]!=200):
        return Response(status_code=dt["status"])
    id=dt["data"]
    for i in meta_data.lst:
        dt=base.add_description(i, id)
        if(dt["status"]!=200):
            return Response(status_code=dt["status"])
    return Response()
@app.get("/sim")
def get_sim(text: str):
    dt=base.get_sim(text)
    if(dt["status"]!=200):
        return Response(
            status_code=dt["status"]
        )
    text_gif, file = dt["data"]
    h=text_gif.encode()
    return Response(
        content=file,
        media_type="application/octet-stream",
        headers={"text_gif": str(h, encoding="Latin-1")}
    )
@app.post("/cnt_desc")
def get_cnt(
    file: UploadFile = File(...)
):
    dt=base.get_id_gif(file.file.read())
    if(dt["status"]!=200):
        return Response(status_code=dt["status"])
    gif_id=dt["data"]
    cnt=base.get_cnt_desc(gif_id)
    return Response(
        headers={"count":str(cnt["data"])}
    )
@app.delete("/del")
def delete_gif(
    file: UploadFile = File(...)
):
    dt=base.get_id_gif(file.file.read())
    if dt["status"]!=200:
        return Response(status_code=404)
    dt=base.del_gif(dt["data"])
    if dt["status"]!=200:
        return Response(status_code=404)
    return Response()
if __name__ == "__main__":
     logging.basicConfig(filename='lg.log')
     import uvicorn
     uvicorn.run(app, host="127.0.0.1", port=80)