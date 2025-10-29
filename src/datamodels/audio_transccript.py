# Хранение результата транскрибации аудио

from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class Segment:
    start: float  
    end: float    
    text: str     

@dataclass(frozen=True)
class Transcript:
    segments: List[Segment]  # список фрагментов
    language: Optional[str] = None  