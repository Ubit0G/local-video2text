# Извлечение кадров из сцен

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np

from src.datamodels.video_transcript import Scene, Picture

class FrameSamplerAbs(ABC):
    @abstractmethod
    def sample_frames(self, video_path: Union[str, Path], scenes: List[Scene]) -> List[Picture]:
        pass

class FrameSampler(FrameSamplerAbs):

    # Извлечение кадра из середины сцены и начала сцены
    def sample_frames(self, video_path: Union[str, Path], scenes: List[Scene]) -> List[Picture]:

        if not scenes:
            return []

        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            sampled_frames: List[Picture] = []

            for scene_id, scene in enumerate(scenes):

                start_time = scene.start_seconds
                start_frame_number = int(start_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
                ret_start, frame_start = cap.read()

                mid_time = (scene.start_seconds + scene.end_seconds) / 2.0
                mid_frame_number = int(mid_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_number)
                ret_mid, frame_mid = cap.read()

                if ret_start:
                    sampled_frames.append(Picture(time = start_time, scene_id = scene_id, picture = frame_start))
                if ret_mid:
                    sampled_frames.append(Picture(time = mid_time, scene_id = scene_id, picture = frame_mid))

            return sampled_frames

        finally:
            cap.release()

    # Сохранение извлеченных кадров (Можно вызвать если будет нужно)
    def save_frames(
        self,
        frames: List[Tuple[float, np.ndarray]],
        output_dir: Union[str, Path],
        filename_prefix: str = "frame",
        image_format: str = "jpg",
    ) -> List[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []

        for i, (timestamp, frame) in enumerate(frames):
            # Форматируем время как 00012_34 (12 секунд, 34 сотых)
            time_str = f"{timestamp:07.2f}".replace(".", "_")
            filename = f"{filename_prefix}_{i:04d}_{time_str}.{image_format}"
            filepath = output_dir / filename

            success = cv2.imwrite(str(filepath), frame)
            saved_paths.append(filepath)

        return saved_paths