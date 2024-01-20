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
from typing import List, Tuple, Literal
import logging
import json
from collections import defaultdict

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


def inline_special_tokens_annotation_to_annotated_document(
    inline_annotation: str,
    entity: str,
    start_pattern: str,
    end_pattern: str,
) -> AnnotatedDocument:
    annotations = set()
    offset = 0
    entities_pattern = [rf"{start_pattern}(.*?){end_pattern}"]
    all_matches = [
        re.finditer(entity_pattern, inline_annotation)
        for entity_pattern in entities_pattern
    ]
    all_matches = [match for matches in all_matches for match in matches]
    # sort all matches by start position in order to change the offset correctly
    all_matches = sorted(all_matches, key=lambda x: x.start())
    for match in all_matches:
        match_offset = len(match.group(0)) - len(match.group(1))
        start = match.start() - offset
        end = match.end() - offset - match_offset
        offset += match_offset
        entity_name = entity
        annotations.add(Annotation(start, end, entity_name, text=match.group(1)))
    for match in all_matches:
        inline_annotation = inline_annotation.replace(match.group(0), match.group(1))
    annotated_document = AnnotatedDocument(
        text=inline_annotation, annotations=annotations
    )
    return annotated_document


def json_annotation_to_annotated_document(
    json_annotation_str: str, entity_set: List[str], original_text: str
) -> AnnotatedDocument:
    text = original_text
    annotations = set()
    try:
        # improve the parsing of the json

        json_annotation = json.loads(json_annotation_str)

        # check if the json have the correct format
        # KEEP IN MIND: MULTIPLE MENTIONS OF THE SAME ENTITY ARE NOT ALLOWED

    except json.decoder.JSONDecodeError:
        logger.warning(
            f"A valid JSON could not be found in the model response: {json_annotation_str}"
        )
        json_annotation = {}

    for entity_name, entity_mentions in json_annotation.items():
        for entity_mention in entity_mentions:
            start = text.find(entity_mention)
            if start != -1 and entity_name in entity_set:
                end = start + len(entity_mention)
                annotations.add(
                    Annotation(start, end, entity_name, text=entity_mention)
                )
    return AnnotatedDocument(text=text, annotations=annotations)


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
        if annotation.text not in original_text:  # type: ignore
            logger.warning(
                f"The text cannot be perfectly aligned: {annotation} was removed because the string is not in the text."
            )
            perfect_align = False
            removed_annotations.append(annotation)
            fixed_annotations_2.remove(annotation)
        elif (annotation.start < 0) | (annotation.end < 0):
            # check if annotation.text is only one time in original_text
            if original_text.count(annotation.text) == 1:  # type: ignore
                # find annotation.text indices in original_text
                start = original_text.find(annotation.text)  # type: ignore
                end = start + len(annotation.text)  # type: ignore
                # update annotation indices in fixed_annotation_2
                for annotation_2 in fixed_annotations_2:
                    if annotation_2 == annotation:
                        annotation_2.start = start
                        annotation_2.end = end
            else:
                logger.warning(
                    f"The text cannot be perfectly aligned: {annotation} was removed because the string was found multiple times."
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
                completion_text=chatgpt_annotated_document.text,
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


def annotated_document_to_json_annotated_string(
    annotated_document: AnnotatedDocument,
):
    annotations = defaultdict(list)
    for annotation in annotated_document.annotations:
        annotations[annotation.label].append(
            annotated_document.text[annotation.start : annotation.end]
        )
    return json.dumps(annotations)


def annotated_document_to_single_turn_few_shot_example(
    annotated_document: AnnotatedDocument,
    answer_shape: Literal["inline", "json"] = "inline",
):
    if answer_shape == "inline":
        annotated_string = annotated_document_to_inline_annotated_string(
            annotated_document
        )
    elif answer_shape == "json":
        annotated_string = annotated_document_to_json_annotated_string(
            annotated_document
        )
    else:
        raise ValueError(
            f"answer_shape should be 'inline' or 'json', but {answer_shape} was given."
        )
    return {"input": annotated_document.text, "output": annotated_string}


def annotated_document_to_multi_turn_chat(
    annotated_document: AnnotatedDocument,
    entity: str,
    parsing_method: str,
    human_msg: str,
):
    if parsing_method == "inline":
        inline_annotated_string = annotated_document_to_inline_annotated_string(
            annotated_document
        )
        return {"input": human_msg, "output": inline_annotated_string}
    elif parsing_method == "json":
        json_annotation = {}
        json_annotation[entity] = [
            annotation.text for annotation in annotated_document.annotations
        ]
        return {"input": human_msg, "output": json.dumps(json_annotation)}

    return {"input": "", "output": ""}


def detokenizer(tokens: List[str]) -> str:
    return twd().detokenize(tokens)


def tokenizer(text: str) -> List[str]:
    return twt().tokenize(text)
