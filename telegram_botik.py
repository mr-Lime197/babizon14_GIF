from main import Gif_added, FileMeta
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sqlalchemy import  create_engine, Column, Integer, String, LargeBinary, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session, relationship
class Base(DeclarativeBase): pass
import logging
from telegram import Bot, Update,ReplyKeyboardMarkup, ReplyKeyboardRemove, File, Message
from telegram.ext import Application, CommandHandler, ContextTypes
import requests
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler
)
ROOT=1
ADD=2
DESC=3
GLOB=4
TOKEN = ""
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(f"Babizon14_bot приветствует тебя, введи описание гифки")
    return GLOB

async def send_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет файл"""
    text = update.message.text
    response = requests.get("http://127.0.0.1:80/sim", params={"text": text})
    text = str(response.headers["text_gif"].encode("Latin-1"), encoding="utf-8")
    byte_sequence = response.content
    await update.message.reply_document(
        document=byte_sequence,
        filename="generated_file.mp4",
        caption=text
    )

async def root(update: Update, context: ContextTypes.DEFAULT_TYPE):
    menu = [["add gif", "cancel"]]
    reply_markup = ReplyKeyboardMarkup(menu, resize_keyboard=True)
    await update.message.reply_text(
        "Ты зашел под правами root",
        reply_markup=reply_markup
    )
    return ROOT

async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text == "cancel":
        await update.message.reply_text(
            "Отменено",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END
    if text == "add gif":
        menu = [["cancel"]]
        await update.message.reply_text(
            "Пришлите гифку в формате mp4",
            reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
        )
        return ADD
async def menu2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text=="cancel":
        menu=[["add gif", "cancel"]]
        reply_markup = ReplyKeyboardMarkup(menu, resize_keyboard=True)
        await update.message.reply_text(
            "отменено",
        )
        await update.message.reply_text(
            "Ты зашел под правами root",
            reply_markup=reply_markup
        )
        return ROOT
    await update.message.reply_text(
        "Неверный формат",
    )
    return ADD

async def gif(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.document:
        file_id = update.message.document.file_id
    elif update.message.video:
        file_id = update.message.video.file_id
    else:
        await update.message.reply_text("пришлите видео в формате MP4")
        return ADD
    
    file = await context.bot.get_file(file_id)
    f = await file.download_as_bytearray()
    context.user_data['gif_data'] = f
    await update.message.reply_text("Введите описание гифки")
    return DESC

async def desc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text=="cancel":
        await update.message.reply_text("Отменено")
        await update.message.reply_text("Пришлите гифку в формате mp4")
        return ADD
    gif_data = context.user_data.get('gif_data')
    if not gif_data:
        await update.message.reply_text("Ошибка: данные гифки потеряны")
        return ConversationHandler.END
    meta = FileMeta(text=str(text))
    response = requests.post(
        url="http://127.0.0.1:80/file/upload-file",
        files={"file": ("animation.mp4", gif_data, "video/mp4")},
        data={"meta": meta.model_dump_json()},
    )
    
    if response.headers["status"] == "0":
        await update.message.reply_text("Гифка успешно добавленна")
    else:
        await update.message.reply_text("ERROR Гифка уже была добавленна ранее")
    
    menu = [["add gif", "cancel"]]
    await update.message.reply_text(
        "Выберите действие:",
        reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
    )
    return ROOT

def main() -> None:
    application = Application.builder().token(TOKEN).build()
    conv_root_handler = ConversationHandler(
        entry_points=[CommandHandler("root", root)],
        states={
            ROOT: [MessageHandler(filters.TEXT & ~filters.COMMAND, menu)],
            ADD: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, menu2),
                MessageHandler(filters.VIDEO | filters.Document.VIDEO, gif)
            ],
            DESC: [MessageHandler(filters.TEXT & ~filters.COMMAND, desc)]
        },
        fallbacks=[],
        allow_reentry=True
    )
    
    gl_conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            GLOB: [
                conv_root_handler,
                MessageHandler(filters.TEXT & ~filters.COMMAND, send_file)
            ]
        },
        fallbacks=[CommandHandler("start", start)],
        allow_reentry=True
    )
    application.add_handler(gl_conv)
    application.run_polling()

if __name__ == "__main__":
    main()