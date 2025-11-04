# Хранение результатов обработки видео

from dataclasses import dataclass

@dataclass(frozen=True)
class Scene:
    start_seconds: float
    end_seconds: float
