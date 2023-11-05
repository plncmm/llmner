from llmner.data import (
    Annotation,
    AnnotatedDocument,
    Conll,
    NotPerfectlyAlignedError,
    AnnotatedDocumentWithException,
)
from difflib import SequenceMatcher
from copy import deepcopy
import re
from nltk.tokenize import TreebankWordTokenizer as twt
from nltk.tokenize.treebank import TreebankWordDetokenizer as twd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def dict_to_enumeration(d: dict) -> str:
    enumeration = ""
    for key, value in d.items():
        enumeration += f"- {key}: {value}\n"
    return enumeration.strip()


def inline_annotation_to_annotated_document(
    inline_annotation: str, entity_set: List[str]
) -> AnnotatedDocument:
    annotations = set()
    offset = 0
    entities_pattern = [rf"<(.*?)>(.*?)</(.*?)>"]
    all_matches = [
        re.finditer(entity_pattern, inline_annotation)
        for entity_pattern in entities_pattern
    ]
    all_matches = [match for matches in all_matches for match in matches]
    # sort all matches by start position in order to change the offset correctly
    all_matches = sorted(all_matches, key=lambda x: x.start())
    for match in all_matches:
        match_offset = len(match.group(0)) - len(match.group(2))
        start = match.start() - offset
        end = match.end() - offset - match_offset
        offset += match_offset
        # getting the entity name
        entity_name = match.group(1)
        # add the entity to the dictionary like this: {entity_name: [ [named_entity,start, end], [named_entity, start, end] , ...]}
        if entity_name in entity_set:
            annotations.add(Annotation(start, end, entity_name, text=match.group(2)))
    for match in all_matches:
        inline_annotation = inline_annotation.replace(match.group(0), match.group(2))
    annotated_document = AnnotatedDocument(
        text=inline_annotation, annotations=annotations
    )
    return annotated_document


def align_annotation(
    original_text: str, chatgpt_annotated_document: AnnotatedDocument
) -> AnnotatedDocument | AnnotatedDocumentWithException:
    fixed_annotation = deepcopy(chatgpt_annotated_document)
    a = chatgpt_annotated_document.text
    b = original_text

    total_difs = [
        (tag, i1, i2, j1, j2, a[i1:i2], b[j1:j2])
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes()
    ]

    replace_difs = [dif for dif in total_difs if dif[0] == "replace"]

    # fix the replace difs
    for dif in replace_difs:
        a = dif[5]
        b = dif[6]
        new_entity_difs = [
            (
                tag,
                i1 + dif[1],
                i2 + dif[1],
                j1 + dif[3],
                j2 + dif[3],
                a[i1:i2],
                b[j1:j2],
            )
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes()
        ]
        total_difs.remove(dif)
        total_difs += new_entity_difs

    for entity in fixed_annotation.annotations:
        difs = [dif for dif in total_difs if dif[1] <= entity.start]
        offset = sum([(dif[4] - dif[3]) - (dif[2] - dif[1]) for dif in difs])
        entity.start += offset
        entity.end += offset

    fixed_annotation.text = original_text

    # remove annotations that not exist in original text
    # because gpt adds or modifies text

    fixed_annotations_2 = list(fixed_annotation.annotations)

    perfect_align = True
    removed_annotations = []
    for annotation in fixed_annotations_2.copy():
        if (
            (annotation.text not in original_text)  # type: ignore
            | (annotation.start < 0)
            | (annotation.end < 0)
        ):
            logger.warning(
                f"The text cannot be perfectly aligned: {annotation} was removed."
            )
            perfect_align = False
            removed_annotations.append(annotation)
            fixed_annotations_2.remove(annotation)

    fixed_annotation.annotations = set(fixed_annotations_2)

    if perfect_align:
        if chatgpt_annotated_document.text != original_text:
            logger.info(
                f"The text was aligned: {chatgpt_annotated_document.text} -> {original_text}"
            )
        return fixed_annotation
    else:
        return AnnotatedDocumentWithException(
            text=fixed_annotation.text,
            annotations=fixed_annotation.annotations,
            exception=NotPerfectlyAlignedError(
                "The text cannot be perfectly aligned",
                removed_annotations=removed_annotations,
                completion_text=chatgpt_annotated_document.text
            ),
        )


def conll_to_inline_annotated_string(conll: List[Tuple[str, str]]) -> str:
    annotated_string = ""
    current_entity = None

    for token, label in conll:
        if label.startswith("B-"):
            if current_entity:
                annotated_string = annotated_string[:-1]
                annotated_string += f"</{current_entity}> "
            entity_class = label[2:]
            annotated_string += f"<{entity_class}>{token}"
            current_entity = entity_class
        elif label.startswith("I-"):
            if current_entity:
                annotated_string += f"{token}"
        else:
            if current_entity:
                annotated_string = annotated_string[:-1]
                annotated_string += f"</{current_entity}> {token}"
                current_entity = None
            else:
                annotated_string += f"{token}"
        annotated_string += " "

    if current_entity:
        annotated_string = annotated_string[:-1]
        annotated_string += f"</{current_entity}>"

    return annotated_string.strip()


def annotated_document_to_conll(
    annotated_document: AnnotatedDocument,
) -> Conll:
    spans = list(twt().span_tokenize(annotated_document.text))
    tokens = [annotated_document.text[span[0] : span[1]] for span in spans]
    boundaries = [span[0] for span in spans]
    conll = ["O"] * len(spans)
    for annotation in annotated_document.annotations:
        start = annotation.start
        end = annotation.end
        label = annotation.label
        start_idx = 0
        for i, boundary in enumerate(boundaries):
            if start == boundary:
                start_idx = i
                break
            elif start < boundary:
                start_idx = i
                break
        end_idx = start_idx
        for i, boundary in list(enumerate(boundaries))[start_idx + 1 :]:
            if end == boundary:
                end_idx = i - 1
                break
            elif end < boundary:
                end_idx = i - 1
                break
        for i in range(start_idx, end_idx + 1):
            if i == start_idx:
                conll[i] = f"B-{label}"
            else:
                conll[i] = f"I-{label}"
    return list(zip(tokens, conll))


def annotated_document_to_inline_annotated_string(
    annotated_document: AnnotatedDocument,
):
    annotated_document = deepcopy(annotated_document)
    inline_annotated_string = annotated_document.text
    annotations = sorted(annotated_document.annotations, key=lambda x: x.start)
    for i in range(len(annotations)):
        annotation = annotations[i]
        start = annotation.start
        end = annotation.end
        label = annotation.label
        text = inline_annotated_string[start:end]
        inline_annotation = f"<{label}>{text}</{label}>"
        inline_annotated_string = (
            inline_annotated_string[:start]
            + inline_annotation
            + inline_annotated_string[end:]
        )
        for j in range(i, len(annotations)):
            annotations[j].start += len(inline_annotation) - len(text)
            annotations[j].end += len(inline_annotation) - len(text)

    return inline_annotated_string


def annotated_document_to_few_shot_example(annotated_document: AnnotatedDocument):
    inline_annotated_string = annotated_document_to_inline_annotated_string(
        annotated_document
    )
    return {"input": annotated_document.text, "output": inline_annotated_string}


def detokenizer(tokens: List[str]) -> str:
    return twd().detokenize(tokens)


def tokenizer(text: str) -> List[str]:
    return twt().tokenize(text)
