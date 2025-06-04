# import telebot;
from main import Gif_added
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sqlalchemy import  create_engine, Column, Integer, String, LargeBinary, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session, relationship
class Base(DeclarativeBase): pass
import logging
from telegram import Bot, Update
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
ASK_TEXT = 1
# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = ""  # Замените на реальный токен

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(f"Babizon14_bot приветствует тебя, введи /gif, затем описание гифки")

async def send_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет файл"""
    text=update.message.text
    response= requests.get("http://127.0.0.1:80/sim", params={"text":text})
    text=str(response.headers["text_gif"].encode("Latin-1"), encoding="utf-8")
    byte_sequence =response.content
    # Отправляем файл как документ
    await update.message.reply_document(
        document=byte_sequence,
        filename="generated_file.mp4",
        caption=text
    )
async def gif_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Запрашивает текст у пользователя"""
    await update.message.reply_text(
        "введите описание гифки из babizon14\n"
        "Для отмены введите /cancel"
    )
    return ASK_TEXT
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отменяет диалог"""
    await update.message.reply_text("Отменено")
    return ConversationHandler.END
def main() -> None:
    application = Application.builder().token(TOKEN).build()
    
    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start))
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("gif", gif_text)],
        states={
            ASK_TEXT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, send_file),
                CommandHandler("cancel", cancel)
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    application.add_handler(conv_handler)
    # Запуск бота
    application.run_polling()

if __name__ == "__main__":
    main()