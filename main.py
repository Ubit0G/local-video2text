import argparse
from pathlib import Path

from src.core.audio_extractor import FfmpegAudioExtractor
from src.core.audio_transcriber import WhisperTranscriber
from src.pipeline import VideoToTextPipeline

def main():
    parser = argparse.ArgumentParser(description="Convert video to text using local Whisper.")
    parser.add_argument("video", type=Path, help="Path to input video")
    parser.add_argument("--model", default="base", help="Whisper model size")
    parser.add_argument("--language", default=None, help="Language code (e.g., 'ru')")
    parser.add_argument("--temp-dir", type=Path, default=Path("data/temp"), help="Temp audio dir")

    args = parser.parse_args()

    extractor = FfmpegAudioExtractor()
    transcriber = WhisperTranscriber(model_size=args.model)
    pipeline = VideoToTextPipeline(
        audio_extractor=extractor,
        transcriber=transcriber,
        temp_dir=args.temp_dir
    )

    transcript = pipeline.run(args.video, language=args.language)

    output_path = Path("data/output") / f"{args.video.stem}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in transcript.segments:
            f.write(f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}\n")
    print(f"✅ Transcription saved to {output_path}")

if __name__ == "__main__":
    main()