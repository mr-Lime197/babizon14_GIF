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
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, ForeignKey, delete, select
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from fastapi.responses import FileResponse
from fastapi.responses import Response
import io
database="sqlite+aiosqlite:///data.db"

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
        self.__tokenizer=AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
        self.__model=AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
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
        self.path = path
        self.model = model
        self.engine = create_async_engine(path)
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
        
    async def init_models(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def add_description(self, txt_desc: str, gif_id: int):
        async with self.async_session() as s:
            async with s.begin():
                result = await s.execute(select(Gif_info_db).where(Gif_info_db.id == gif_id))
                if result.scalar_one_or_none() is None:
                    logging.error(f"gif with id {gif_id} is not exist")
                    return {"status":404}
                
                gif_emb = Gif_emb_db(
                    description=txt_desc, 
                    embeding=self.model.get_emb(txt_desc),
                    id_gif=gif_id
                )
                s.add(gif_emb)
                logging.info(f"description {txt_desc} add for gif {gif_id}")
        return {"status":200}
    
    async def add_gif(self, Gif:Gif_added) -> dict:
        async with self.async_session() as s:
            async with s.begin():
                result = await s.execute(select(Gif_info_db).where(Gif_info_db.file == Gif.video))
                if result.scalar_one_or_none():
                    logging.error(f'GIFs: {Gif.text} is exist')
                    return {"status":404}
                
                gif = Gif_info_db(text=Gif.text, file=Gif.video)
                s.add(gif)
                await s.flush()  # Получаем ID перед коммитом
                logging.info(f"added gif {Gif.text}")
                await self.add_description(txt_desc=gif.text, gif_id=gif.id)
        return {"status":200}
    
    async def get_id_gif(self, file:bytes) -> dict:
        async with self.async_session() as s:
            result = await s.execute(select(Gif_info_db).where(Gif_info_db.file == file))
            gif = result.scalar_one_or_none()
            if gif is None:
                logging.error("gif is not exist")
                return {"status":404, "data":None}
            return {"status":200, "data":gif.id}
    
    async def get_sim(self, text: str) -> dict:
        model=self.model
        bestsim=0
        bestid=-1
        emb=np.frombuffer(model.get_emb(text), dtype=np.float32)
        async with self.async_session() as s:
            result = await s.execute(select(Gif_emb_db))
            all_embs = result.scalars().all()
            
            if not all_embs:
                logging.error("database is empty")
                return {"status":500, "data":None}
            
            for row in all_embs:
                curemb=np.frombuffer(row.embeding, dtype=np.float32)
                sim=model.dist(emb, curemb)
                if(sim<bestsim or bestid==-1):
                    bestsim=sim
                    bestid=row.id_gif
            
            gif_info = await s.get(Gif_info_db, bestid)
            return {"status":200, "data":(gif_info.text, gif_info.file)}
    
    async def get_cnt_desc(self, gif_id: int) -> dict:
        async with self.async_session() as s:
            result = await s.execute(select(Gif_emb_db).where(Gif_emb_db.id_gif == gif_id))
            count = len(result.scalars().all())
            return {"status":200, "data":count}
    
    async def del_gif(self, gif_id:int) ->dict:
        cnt = await self.get_cnt_desc(gif_id)
        if cnt["status"] == 200 and cnt["data"] > 0:
            async with self.async_session() as s:
                async with s.begin():
                    await s.execute(delete(Gif_info_db).where(Gif_info_db.id == gif_id))
                    await s.execute(delete(Gif_emb_db).where(Gif_emb_db.id_gif == gif_id))
            return {"status":200}
        return {"status":404}

app = FastAPI()
model=Model()
base=db(database, model)
base.init_models()

@app.post("/file/upload-file")
async def upload_file_bytes(
    file: UploadFile = File(...),
    meta: str = Form(...),
):
    meta_data = FileMeta(**json.loads(meta))
    video_bytes = await file.read()
    status = await base.add_gif(Gif_added(text=meta_data.text, video=video_bytes))
    return Response(status_code=status["status"])

@app.post("/file/add_desc")
async def add_desc(
    file: UploadFile = File(...),
    meta: str = Form(...),
):
    meta_data = Desc(**json.loads(meta))
    video_bytes = await file.read()
    dt = await base.get_id_gif(video_bytes)
    if(dt["status"]!=200):
        return Response(status_code=dt["status"])
    id=dt["data"]
    for i in meta_data.lst:
        dt = await base.add_description(i, id)
        if(dt["status"]!=200):
            return Response(status_code=dt["status"])
    return Response()

@app.get("/sim")
async def get_sim(text: str):
    dt = await base.get_sim(text)
    if(dt["status"]!=200):
        return Response(status_code=dt["status"])
    text_gif, file = dt["data"]
    h = text_gif.encode()
    return Response(
        content=file,
        media_type="application/octet-stream",
        headers={"text_gif": str(h, encoding="Latin-1")}
    )

@app.post("/cnt_desc")
async def get_cnt(
    file: UploadFile = File(...)
):
    video_bytes = await file.read()
    dt = await base.get_id_gif(video_bytes)
    if dt["status"] != 200:
        return Response(status_code=dt["status"])
    gif_id = dt["data"]
    cnt = await base.get_cnt_desc(gif_id)
    return Response(headers={"count": str(cnt["data"])})

@app.delete("/del")
async def delete_gif(
    file: UploadFile = File(...)
):
    video_bytes = await file.read()
    dt = await base.get_id_gif(video_bytes)
    if dt["status"] != 200:
        return Response(status_code=404)
    dt = await base.del_gif(dt["data"])
    if dt["status"] != 200:
        return Response(status_code=404)
    return Response()

if __name__ == "__main__":
    logging.basicConfig(filename='lg.log')
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=80)