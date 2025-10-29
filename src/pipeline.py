# Управление процессом преобразования видео в текст

from pathlib import Path
from typing import Optional

from src.core.audio_extractor import AudioExtractor
from src.core.audio_transcriber import Transcriber
from src.datamodels.audio_transccript import Transcript

class VideoToTextPipeline:
    def __init__(
        self,
        audio_extractor: AudioExtractor,
        transcriber: Transcriber,
        temp_dir: Path = Path("data/temp")
    ):
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)    

    def run(self, video_path: Path, language: Optional[str] = None) -> Transcript:
        audio_path = self.temp_dir / f"{video_path.stem}.wav"
        self.audio_extractor.extract(video_path, audio_path)
        transcript = self.transcriber.transcribe(audio_path, language=language)
        
        return transcript