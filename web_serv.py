import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel,  BertTokenizer, BertModel
from pydantic import BaseModel
import pydantic
import torch
import json
import logging
from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import  create_engine, Column, Integer, String, LargeBinary, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session, relationship
import requests
app = FastAPI()

# Модель для метаданных файла
class FileMeta(BaseModel):
    text: str

@app.post("/upload_to_server/")
def upload_to_server(
    file: UploadFile = File(...),
    text: str = Form(...),
):
    meta = FileMeta(text=text)
    response=requests.post(
        url="http://127.0.0.1:80/file/upload-file",
        files = {"file": (file.filename, file.file.read(), file.content_type)},
        data = {"meta": meta.model_dump_json()},
    )
if __name__ == "__main__":
     import uvicorn 
     uvicorn.run(app, host="127.0.0.1", port=8080)


