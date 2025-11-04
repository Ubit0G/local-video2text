# Управление процессом преобразования видео в текст

from pathlib import Path
from typing import Optional

from src.core.audio_processor.audio_extractor import AudioExtractor
from src.core.audio_processor.audio_transcriber import Transcriber
from src.core.video_processor.scene_detector import SceneDetectionAbs
from src.core.video_processor.frame_sampler import FrameSamplerAbs
from src.core.video_processor.frame_processor import FrameProcessorAbs
from src.datamodels.audio_transcript import Transcript
from src.utils.io import save_results_to_json

class VideoToTextPipeline:
    def __init__(
        self,
        audio_extractor: AudioExtractor,
        transcriber: Transcriber,
        scenedetector: SceneDetectionAbs,
        framesampler: FrameSamplerAbs,
        frameprocessor: FrameProcessorAbs,
        temp_dir: Path = Path("data/temp")
    ):
        self.audio_extractor = audio_extractor
        self.transcriber = transcriber
        self.scenedetector  = scenedetector
        self.framesampler = framesampler
        self.frameprocessor = frameprocessor
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)    

    def run(self, video_path: Path, audiolanguage: Optional[str] = None, output_dir = Path("data/output")):
        audio_path = self.temp_dir / f"{video_path.stem}.wav"
        self.audio_extractor.extract(video_path, audio_path)
        transcript = self.transcriber.transcribe(audio_path, language=audiolanguage)
        scenes = self.scenedetector.process(video_path)
        frames = self.framesampler.sample_frames(video_path, scenes)
        processed = self.frameprocessor.process_frames(frames)
        save_results_to_json(transcript=transcript, video_results=processed, output_dir=output_dir)
