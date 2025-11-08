# Обработка полученных сцен 

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import torch
import easyocr
from transformers import AutoProcessor, AutoModelForCausalLM

from src.datamodels.video_transcript import Picture

logger = logging.getLogger(__name__)

class FrameProcessorAbs(ABC):
    @abstractmethod
    def process_frames(self, frames: List[Picture]) -> List[Tuple[float, int, str, List[str]]]:
        pass

class FrameProcessor(FrameProcessorAbs):

    def __init__(
        self,
        florence_model_name: str = "microsoft/Florence-2-base", # Имя модели Florence-2 на Hugging Face.
        ocr_languages: List[str] = None, # Список языков для EasyOCR (например, ["ru", "en"]).
        device: Optional[str] = None, # Устройство ("cuda", "cpu"). Если None — автоопределение.
        use_half: bool = True, # Использовать float16 на GPU
        max_new_tokens: int = 150, # Максимальная длина генерируемого caption
    ):
        if ocr_languages is None:
            ocr_languages = ["ru", "en"]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = use_half and (self.device == "cuda")
        self.max_new_tokens = max_new_tokens
        self.ocr_languages = ocr_languages

        # Загрузка Florence-2
        logger.info(f"Загрузка Florence-2: {florence_model_name} на {self.device}")
        self.florence_processor = AutoProcessor.from_pretrained(
            florence_model_name, trust_remote_code=True
        )
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            florence_model_name,
            trust_remote_code=True,
            dtype=torch.float16 if self.use_half else torch.float32,
            attn_implementation="eager"
        ).to(self.device)
        self.florence_model.eval()

        # Загрузка EasyOCR
        logger.info(f"Загрузка EasyOCR для языков: {ocr_languages} (GPU={self.device == 'cuda'})")
        self.ocr_reader = easyocr.Reader(ocr_languages, gpu=(self.device == "cuda"))

    # Возвращает список кортежей: (таймкод, caption, [on-screen текст]).
    def process_frames(self,frames: List[Picture]) -> List[Tuple[float, int, str, List[str]]]:

        if not frames:
            return []

        results = []
        for frame in frames:
            if frame.picture is None or not isinstance(frame.picture, np.ndarray) or frame.picture.size == 0:
                continue

            # Caption через Florence-2
            caption = self._generate_caption(frame.picture)

            # OCR через EasyOCR
            ocr_texts = self._run_ocr(frame.picture)

            results.append((frame.time, frame.scene_id, caption, ocr_texts))

        return results

    # Генерирует описание кадра через Florence-2
    def _generate_caption(self, frame_bgr: np.ndarray) -> str:
        """
        Генерирует детализированное описание кадра через Florence-2.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        prompt = "<CAPTION>" # Можно изменять этот промт для детализации

        inputs = self.florence_processor(
            text=prompt, images=pil_image, return_tensors="pt"
        ).to(self.device)

        if self.use_half:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                num_beams=3,
                do_sample=False,
                use_cache=False
            )

        # Декодируем и чистим служебные токены
        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        caption = (
            generated_text
            .replace("<s>", "")
            .replace("</s>", "")
            .strip()
        )
        return caption

    # Распознаёт текст на кадре через EasyOCR
    def _run_ocr(self, frame_bgr: np.ndarray) -> List[str]:
        """
        Распознаёт текст на кадре через EasyOCR.
        """
        results = self.ocr_reader.readtext(frame_bgr)
        return [text.strip() for (bbox, text, conf) in results if conf >= 0.5]