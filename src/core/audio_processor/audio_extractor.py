# Извлечение аудио из видео

import abc
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

class AudioExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, video_path: Union[str, Path], output_audio_path: Union[str, Path]) -> Path:
        pass

# Извлеечние при помощи ffmpeg
class FfmpegAudioExtractor(AudioExtractor):

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate # Частота
        self.channels = channels # Кол-во аудиоканалов

    def extract(self, video_path: Union[str, Path], output_audio_path:  Union[str, Path]) -> Path:
        from pydub import AudioSegment
        video_path = Path(video_path)
        output_audio_path = Path(output_audio_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        audio = AudioSegment.from_file(video_path)
        audio = audio.set_frame_rate(self.sample_rate).set_channels(self.channels)
        audio.export(output_audio_path, format="wav")
        
        logger.info(f"Аудио успешно извлечено и сохранено: {output_audio_path}")
        return output_audio_path