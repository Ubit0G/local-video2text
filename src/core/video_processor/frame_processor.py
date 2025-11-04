# Обработка полученных сцен 

import cv2
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration

class FrameProcessorAbs(ABC):
    @abstractmethod
    def process_frames(self, frames: List[Tuple[float, np.ndarray]]) -> List[Tuple[float, str, List[str]]]:
        pass

class FrameProcessor(FrameProcessorAbs):

    def __init__(
        self,
        blip_model_name: str = "Salesforce/blip-image-captioning-base", # Имя модели BLIP на Hugging Face.
        ocr_languages: List[str] = None, # Список языков для EasyOCR (например, ["ru", "en"]).
        device: Optional[str] = None, # Устройство ("cuda", "cpu"). Если None — автоопределение.
        use_half: bool = True # Использовать float16 на GPU
    ):

        if ocr_languages is None:
            ocr_languages = ["ru", "en"]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = use_half and (self.device == "cuda")

        # Загрузка BLIP
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name, use_fast=True)
        dtype = torch.float16 if self.use_half else torch.float32
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_model_name,
            dtype=dtype
        ).to(self.device)
        self.blip_model.eval()

        # Загрузка EasyOCR 
        self.ocr_reader = easyocr.Reader(ocr_languages, gpu=(self.device == "cuda"))

    # Возвращает список кортежей: (таймкод, caption, [on-screen текст]).
    def process_frames(
        self,
        frames: List[Tuple[float, np.ndarray]]
    ) -> List[Tuple[float, str, List[str]]]:

        if not frames:
            return []

        results = []
        for i, (timestamp, frame_bgr) in enumerate(frames):
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            caption = self._generate_caption(pil_image)

            ocr_texts = self._run_ocr(frame_bgr)

            results.append((timestamp, caption, ocr_texts))

        return results

    # Генерирует описание кадра через BLIP
    def _generate_caption(self, image: Image.Image) -> str:
   
        inputs = self.blip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.use_half:
            inputs = {k: v.half() for k, v in inputs.items()}

        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_new_tokens=50)

        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    # Распознаёт текст на кадре через EasyOCR
    def _run_ocr(self, frame_bgr: np.ndarray) -> List[str]:
        
        results = self.ocr_reader.readtext(frame_bgr)
        # Фильтрация по confidence и извлечение текста
        return [text.strip() for (bbox, text, conf) in results if conf >= 0.5]