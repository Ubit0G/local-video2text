import argparse
from pathlib import Path

from src.core.audio_processor.audio_extractor import FfmpegAudioExtractor
from src.core.audio_processor.audio_transcriber import WhisperTranscriber
from src.pipeline import VideoToTextPipeline
from src.core.video_processor.scene_detector import SceneDetectionProcessor
from src.core.video_processor.frame_sampler import FrameSampler
from src.core.video_processor.frame_processor import FrameProcessor

import logging
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Convert video to text")
    parser.add_argument("video", type=Path, help="Path to input video")

    args = parser.parse_args()

    audio_extractor = FfmpegAudioExtractor()
    transcriber = WhisperTranscriber()
    scenedetector = SceneDetectionProcessor(threshold=28.0, min_scene_duration_sec=2.0)
    framesampler = FrameSampler()
    frameprocessor = FrameProcessor(ocr_languages=["ru", "en"])

    pipeline = VideoToTextPipeline(
        audio_extractor = audio_extractor,
        transcriber = transcriber,
        scenedetector  = scenedetector,
        framesampler = framesampler,
        frameprocessor = frameprocessor
    )

    pipeline.run(args.video)


if __name__ == "__main__":
    main()