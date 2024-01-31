from dataclasses import dataclass
from typing import Set, Optional, List, Tuple, Literal, Union


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


class NotPerfectlyAlignedError(Exception):
    """Exception raised when the text cannot be perfectly aligned.
    Args:
        message (str): Message of the exception.
        removed_annotations (List[Annotation]): List of annotations that were removed.
    """

    def __init__(
        self,
        message: str,
        removed_annotations: List[Annotation] = [],
        completion_text: str = "",
    ):
        self.removed_annotations = removed_annotations
        self.message = message
        self.completion_text = completion_text
        super().__init__(self.message)


Token = str
Label = str
Conll = List[Tuple[Token, Label]]


@dataclass
class PromptTemplate:
    """PromptTemplate class. Used to represent a prompt template.
    Args:
        inline_single_turn (str): Template for inline single turn.
        inline_multi_turn_default_delimiters (str): Template for inline multi turn with default delimiters.
        inline_multi_turn_custom_delimiters (str): Template for inline multi turn with custom delimiters.
        json_single_turn (str): Template for json single turn.
        json_multi_turn (str): Template for json multi turn.
        multi_turn_prefix (str): Prefix for multi turn.
        pos (str): Template for part of speech tagging.
    """

    inline_single_turn: str
    inline_multi_turn_default_delimiters: str
    inline_multi_turn_custom_delimiters: str
    json_single_turn: str
    json_multi_turn: str
    multi_turn_prefix: str
    pos: str
    pos_answer_prefix: str
