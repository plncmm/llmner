from dataclasses import dataclass
from typing import Set, Optional, List, Tuple


@dataclass
class Document:
    """Document class. Used to represent a document.
    Args:
        text (str): Text of the document.
    """

    text: str


@dataclass()
class Annotation:
    """Annotation class. Used to represent an annotation. An annotation is a labelled span of text.
    Args:
        start (int): Start index of the annotation.
        end (int): End index of the annotation.
        label (str): Label of the annotation.
        text (Optional[str], optional): Text content of the annotation. Is is optional and defaults to None.
    """

    start: int
    end: int
    label: str
    text: Optional[str] = None

    def __hash__(self):
        return hash((self.start, self.end, self.label))


@dataclass
class AnnotatedDocument(Document):
    """AnnotatedDocument class. Used to represent an annotated document.
    Args:
        text (str): Text of the document.
        annotations (Set[Annotation]): Set of annotations of the document.
    """

    annotations: Set[Annotation]


@dataclass
class AnnotatedDocumentWithException(AnnotatedDocument):
    """AnnotatedDocumentWithException class. Used to represent an annotated document with an exception.
    Args:
        text (str): Text of the document.
        annotations (Set[Annotation]): Set of annotations of the document.
        exception (Exception): Exception of the document.
    """

    exception: Exception


class NotContextualizedError(Exception):
    pass

Conll = List[Tuple[str, str]]