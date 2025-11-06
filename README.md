# local-video2text
Сервис для генерации текстовых описаний видео с использованием локальных моделей

## Требования
1) Python 3.12.x
2) Установить ffmpeg https://ffmpeg.org/
3) CUDA 13.0

## Запуск
0) `py -3.12  -m venv venv`, `.\venv\Scripts\activate` 
1) `pip install -r requirements.txt`
2) `python main.py "videopath"` 

## Вывод
Два .json файла с аудио транскрибцией и описанием сцен видео

## Пайплайн
1) Извлечение аудио из видео в `audio_extractor` при помощи `ffmpeg`
2) Транскрибация аудио в `audio_transcriber` при помощи `openai-whisper turbo`
3) Разделение видео на сцены в `scene_detector` при помощи `scenedetect`
4) Извлечение кадров из сцен в `frame_sampler`
5) Описание полученных кадров и текста на кадрах в `frame_processor` при помощи `EasyOCR` и `Florence-2`

## Florence-2
Есть возможность использовать разные промты в `prompt`:
1) `<CAPTION>`
2) `<DETAILED_CAPTION>`
3) `<MORE_DETAILED_CAPTION>`

Также есть ограничение количества выходных токенов в `max_new_tokens`

# ToDo:
1) Подумать над выделением кадров из сцен, пока берется один кадр в середине сцены. Возможно добавить 
2) Потестить microsoft/Florence-2-large, пока стоит microsoft/Florence-2-base
3) Потестить варианты промтов и количество токенов для вижн модели
4) Есть проблемы с выделением сцен из коротких видео
5) Подумать над структурой .json файлов
6) Потестить fp16=False/True и use_half: bool = True/False уже на видюхе,пока установлены варианты рабочие на моей видюхе 