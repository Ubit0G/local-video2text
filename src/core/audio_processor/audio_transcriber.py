# Распознание речи из аудиофайла

import abc
import logging
from pathlib import Path
from typing import Optional, Union

from src.datamodels.audio_transcript import Transcript, Segment

logger = logging.getLogger(__name__)

class Transcriber(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, audio_path: Union[str, Path], language: Optional[str] = None) -> Transcript:
        pass

# Распознование при помощи openai-whisper
class WhisperTranscriber(Transcriber):
    def __init__(self, model_size: str = "turbo", device: str = "auto"):
        import whisper
        if whisper.torch.cuda.is_available():
            device = "cuda" 
        else :
            device = "cpu"
        logger.info(f"Загрузка Whisper: {model_size} на {device}")
        self.model = whisper.load_model(model_size)
        self.device = device

    def transcribe(self, audio_path: Union[str, Path], language: Optional[str] = None) -> Transcript:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info("Транскрибция аудио Whisper...")

        # Вызов Whisper
        result = self.model.transcribe(str(audio_path), language=language, fp16=False)

        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in result["segments"]
        ]
        
        logger.info(f"Получена транскрибция аудио на {result.get("language")} языке")
        return Transcript(segments=segments, language=result.get("language"))