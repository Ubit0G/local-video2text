# Распознание речи из аудиофайла

import abc
import abc
from pathlib import Path
from typing import Optional, Union

from src.datamodels.audio_transccript import Transcript, Segment

class Transcriber(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, audio_path: Union[str, Path], language: Optional[str] = None) -> Transcript:
        pass

# Распознование при помощи openai-whisper
class WhisperTranscriber(Transcriber):
    def __init__(self, model_size: str = "base", device: str = "auto"):
        import whisper
        if device == "auto":
            device = "cuda" if whisper.torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size)
        self.device = device

    def transcribe(self, audio_path: Union[str, Path], language: Optional[str] = None) -> Transcript:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Вызов Whisper
        result = self.model.transcribe(str(audio_path), language=language, verbose=True)

        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in result["segments"]
        ]
        
        return Transcript(segments=segments, language=result.get("language"))