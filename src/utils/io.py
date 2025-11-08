# Сохранение результатов в json

import json
from pathlib import Path
from typing import List, Tuple, Optional, Union
from dataclasses import asdict
from src.datamodels.audio_transcript import Transcript, Segment
from src.datamodels.video_transcript import Picture

def save_results_to_json(
    transcript: Transcript,
    video_results: List[Tuple[float, int, str, List[str]]],
    output_dir: Union[str, Path] = "output"
):
    """
    Сохраняет транскрипцию и результаты обработки видео в JSON-файлы.

    Args:
        transcript: Объект Transcript с аудиосегментами.
        video_results: Список (таймкод, caption, [on-screen текст]).
        output_dir: Директория для сохранения файлов.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Сохраняем транскрипцию
    transcript_dict = {
        "language": transcript.language,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            }
            for seg in transcript.segments
        ]
    }
    with open(output_path / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript_dict, f, ensure_ascii=False, indent=2)

    # 2. Сохраняем описание видео
    video_description = [
        {
            "timestamp_seconds": time,
            "scene_id": scene_id,
            "visual_caption": caption,
            "on_screen_text": texts
        }
        for (time, scene_id, caption, texts) in video_results
    ]
    with open(output_path / "video_description.json", "w", encoding="utf-8") as f:
        json.dump(video_description, f, ensure_ascii=False, indent=2)
