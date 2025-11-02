# local-video2text
Сервис для генерации текстовых описаний видео с использованием локальных моделей

## Требования
1) Python 3.12.x
2) Установить ffmpeg https://ffmpeg.org/

## Запуск
0) py -3.12  -m venv venv, .\venv\Scripts\activate 
1) pip install -r requirements.txt для CPU, pip install -r requirementsGPU.txt для GPU
2) python main.py "video" "--model" "--language" "--temp-dir"

## Модели
1) openai-whisper, оптимальная turbo