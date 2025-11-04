# Деление видео на сцены

from abc import ABC, abstractmethod
from typing import List, Union, Optional
from pathlib import Path
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from src.datamodels.video_transcript import Scene

class SceneDetectionAbs(ABC):
    @abstractmethod
    def process(self, video_path: Union[str, Path]) -> List[Scene]:
        pass

# Использование scenedetect
class SceneDetectionProcessor(SceneDetectionAbs):

    def __init__(
        self,
        threshold: float = 27.0, # Порог чувствительности к изменению сцен
        min_scene_duration_sec: Optional[float] = None, # Минимальная продолжительность сцены
    ):
        self.threshold = threshold
        self.min_scene_duration_sec = min_scene_duration_sec

    def process(self, video_path: Union[str, Path]) -> List[Scene]:

        video_path = Path(video_path).resolve()

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video = open_video(
            str(video_path),
        )

        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))

        scene_manager.detect_scenes(video, show_progress=False)

        raw_scene_list = scene_manager.get_scene_list()
        scenes = [
            Scene(
                start_seconds=scene[0].get_seconds(),
                end_seconds=scene[1].get_seconds()
            )
            for scene in raw_scene_list
        ]

        if self.min_scene_duration_sec is not None:
            scenes = self._merge_short_scenes(scenes)

        return scenes

    # Постобработка коротких сцен
    def _merge_short_scenes(self, scenes: List[Scene]) -> List[Scene]:
        if len(scenes) <= 1:
            return scenes

        merged = []
        current = scenes[0]

        for next_scene in scenes[1:]:
            if (current.end_seconds - current.start_seconds) < self.min_scene_duration_sec:
                current = Scene(current.start_seconds, next_scene.end_seconds)
            else:
                merged.append(current)
                current = next_scene

        if (current.end_seconds - current.start_seconds) < self.min_scene_duration_sec and merged:
            last = merged[-1]
            merged[-1] = Scene(last.start_seconds, current.end_seconds)
        else:
            merged.append(current)

        return merged