# local-video2text
Сервис для генерации текстовых описаний видео с использованием локальных моделей

## Требования
1) Python 3.12.x
2) Установить ffmpeg https://ffmpeg.org/
3) CUDA 13.0

## Запуск
0) py -3.12  -m venv venv, .\venv\Scripts\activate 
1) pip install -r requirements.txt 
2) python main.py "videopath" 

## Вывод
Два .json файла с аудио транскрибцией и описанием сцен видео

## Пайплайн
1) Извлечение аудио из видео в `audio_extractor` при помощи `ffmpeg`
2) Транскрибация аудио в `audio_transcriber` при помощи `openai-whisper turbo`
3) Разделение видео на сцены в `scene_detector` при помощи `scenedetect`
4) Извлечение кадров из сцен в `frame_sampler`
5) Описание полученных кадров и текста на кадрах в `frame_processor` при помощи `EasyOCR` и `BLIP-2`

# ToDo:
1) Подумать над выделением кадров из сцен, пока берется один кадр в середине сцены. Возможно добавить 
2) Потестить blip-image-captioning-large, пока стоит blip-image-captioning-base
3) Есть проблемы с выделением сцен из коротких видео
4) Подумать над структурой .json файлов
5) Потестить fp16=False/True и use_half: bool = True/False уже на видюхе,пока установлены варианты рабочие на моей видюхе 