from main import Gif_added, FileMeta, Desc
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
DESC2=4
DEL=5
GLOB=6
TOKEN = ""
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    user = update.effective_user
    await update.message.reply_text(
        text=f"Привет, устал искать подходящий гиф в канале <a href='https://t.me/babizon14'> Овсе гифы</a>?,тогда тебе сюда!\n Просто опиши гиф который ищешь",
        parse_mode="HTML",
        reply_markup=ReplyKeyboardRemove()
        )
    return GLOB

async def send_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет файл"""
    text = update.message.text
    response = requests.get("http://127.0.0.1:80/sim", params={"text": text})
    if response.status_code!=200:
        await update.message.reply_text(
            "Ошибка: База данных пуста"
        )
        return GLOB
    text = str(response.headers["text_gif"].encode("Latin-1"), encoding="utf-8")
    byte_sequence = response.content
    await update.message.reply_document(
        document=byte_sequence,
        filename="generated_file.mp4",
        caption=text
    )

async def root(update: Update, context: ContextTypes.DEFAULT_TYPE):
    menu = [["Добавить новую гифку", "Удалить гифку"],
            ["Назад"]]
    reply_markup = ReplyKeyboardMarkup(menu, resize_keyboard=True)
    await update.message.reply_text(
        "Выполнен вход с правами root",
        reply_markup=reply_markup
    )
    return ROOT

async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text == "Назад":
        await update.message.reply_text(
            "выход из root",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END
    if text == "Добавить новую гифку":
        menu = [["Назад"]]
        await update.message.reply_text(
            "Отправь гифку",
            reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
        )
        return ADD
    if text=="Удалить гифку":
        menu=[["Назад"]]
        await update.message.reply_text(
            text="Отправьте гифку, которую хотите удалить",
            reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
        )
        return DEL
async def menu2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text=="Назад":
        menu = [["Добавить новую гифку", "Удалить гифку"],
            ["Назад"]]
        reply_markup = ReplyKeyboardMarkup(menu, resize_keyboard=True)
        # await update.message.reply_text(
        #     "отменено",
        # )
        await update.message.reply_text(
            "Выполнен вход с правами root",
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
        await update.message.reply_text("Отправь гифку")
        return ADD
    file = await context.bot.get_file(file_id)
    f = await file.download_as_bytearray()
    context.user_data['gif_data'] = f
    await update.message.reply_text("Введите название гифки")
    return DESC

async def desc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text=="Назад":
        #await update.message.reply_text("Отменено")
        await update.message.reply_text("Отправь гифку")
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
    response2 = requests.post(
        url="http://127.0.0.1:80/cnt_desc",
        files={"file": ("animation.mp4", gif_data, "video/mp4")},
    )
    if response.status_code == 200 or (response2.status_code==200 and int(response2.headers["count"])<2):
        menu=[["Назад"]]
        await update.message.reply_text(
            text="Отлично, теперь добавь от 1 до 3 описаний этой гифки\nэто необходимо для лучшего поиска\n",
            reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
            )
        context.user_data["desc"]=[]
        return DESC2
    if response2.status_code!=200:
        await update.message.reply_text("Ошибка: Неверный формат гифки")
    await update.message.reply_text("Ошибка: Гифка уже была добавленна ранее")
    menu = [["Добавить новую гифку", "Удалить гифку"],
            ["Назад"]]
    await update.message.reply_text(
        "Выберите действие:",
        reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
    )
    return ROOT
async def desc2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text=update.message.text
    if text=="Подтвердить":
        gif_data = context.user_data.get('gif_data')
        lst=context.user_data.get("desc")
        if not gif_data:
            await update.message.reply_text("Ошибка: данные гифки потеряны")
            return ConversationHandler.END
        meta = Desc(lst=lst)
        response = requests.post(
            url="http://127.0.0.1:80/file/add_desc",
            files={"file": ("animation.mp4", gif_data, "video/mp4")},
            data={"meta": meta.model_dump_json()},
        )
        if response.status_code!=200:
            await update.message.reply_text(
                "Ошибка: Ошибка на стороне сервера"
            )
            return ConversationHandler.END
        menu = [["Добавить новую гифку", "Удалить гифку"],
            ["Назад"]]
        await update.message.reply_text(
            text="Описания успешно добавлены",
            reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
        )
        return ROOT
    if text=="Удалить последнее описание":
        if not (context.user_data["desc"]) or context.user_data["desc"]==[]:
            await update.message.reply_text(
                "Ошибка: Еще не было добавленно ни одного описания"
            )
            return DESC2
        context.user_data["desc"].pop()
        await update.message.reply_text(
            "Поледнее описание удалено"
        )
        if len(context.user_data["desc"])==0:
            menu=[["Назад"]]
            await update.message.reply_text(
                text=f'Добавлено описаний{len(context.user_data["desc"])}/3',
                reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
            )
        return DESC2
    if text=="Назад":
        menu=[["Назад"]]
        await update.message.reply_text(
            text="Введите название гифки",
            reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
            )
        return DESC
    if len(context.user_data["desc"])==3:
        await update.message.reply_text(
            text=f'Правышено ограничение на кол-во описаний',
        )
        return DESC2
    context.user_data["desc"].append(text)
    menu=[["Подтвердить"], ["Удалить последнее описание"]]
    await update.message.reply_text(
        text=f'Добавлено описаний{len(context.user_data["desc"])}/3',
        reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
    )
    return DESC2
async def del_gif(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.document:
        file_id = update.message.document.file_id
    elif update.message.video:
        file_id = update.message.video.file_id
    else:
        await update.message.reply_text("Отправь гифку")
        return DEL
    file = await context.bot.get_file(file_id)
    f = await file.download_as_bytearray()
    response=requests.delete(
        url="http://127.0.0.1:80/del",
        files={"file": ("animation.mp4", f, "video/mp4")},
    )
    menu=[["Добавить новую гифку", "Удалить гифку"],
            ["Назад"]]
    if response.status_code==200:
        await update.message.reply_text(
            text="Гифка успешно удалена",
            reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
        )
    else:
        await update.message.reply_text(
            text="Ошибка: Гифки не существует",
            reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
        )
    return ROOT
async def del_gif_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text=update.message.text
    if text=="Назад":
        menu=[["Добавить новую гифку", "Удалить гифку"],
            ["Назад"]]
        await update.message.reply_text(
            text="Отменено",
            reply_markup=ReplyKeyboardMarkup(menu, resize_keyboard=True)
        )
        return ROOT
    return DEL
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
            DESC: [MessageHandler(filters.TEXT & ~filters.COMMAND, desc)],
            DESC2:[MessageHandler(filters.TEXT & ~filters.COMMAND, desc2)],
            DEL:[
                MessageHandler(filters.VIDEO | filters.Document.VIDEO, del_gif),
                MessageHandler(filters.TEXT & ~filters.COMMAND, del_gif_cancel)
                ]
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