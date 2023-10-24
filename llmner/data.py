from dataclasses import dataclass
from typing import Set, Optional

@dataclass
class Document:
    text: str


@dataclass()
class Annotation:
    start: int
    end: int
    label: str
    text: Optional[str] = None

    def __hash__(self):
        return hash((self.start, self.end, self.label))


@dataclass
class AnnotatedDocument(Document):
    annotations: Set[Annotation]