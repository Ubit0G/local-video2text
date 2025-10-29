# local-video2text
Сервис для генерации текстовых описаний видео с использованием локальных моделей

## Требования
1) Python 3.12.x
2) Установить ffmpeg https://ffmpeg.org/
3) Пока лично у меня модель не подгружает веса, поэтому скачал отсюда https://huggingface.co/mingu4969/windows-archive-dist/blob/main/whisper/base.pt

## Запуск
0) py -3.12  -m venv venv, .\venv\Scripts\activate 
1) pip install -r requirements.txt
2) python main.py data/(Адрес видео)

## Модели
1) Пока openai-whisper, в последствии на faster-whisper