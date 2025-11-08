# Хранение результатов обработки видео

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Scene:
    start_seconds: float
    end_seconds: float

@dataclass(frozen=True)
class Picture:
    time: float
    scene_id: int
    picture: np.ndarray