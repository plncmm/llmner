from data import Annotation, AnnotatedDocument
from difflib import SequenceMatcher
from copy import deepcopy
import re
from nltk.tokenize import TreebankWordTokenizer as twt
from typing import List

def dict_to_enumeration(d):
    enumeration = ""
    for key, value in d.items():
        enumeration += f"- {key}: {value}\n"
    return enumeration.strip()

def extract_gpt_annotation(
    annotated_text: str, entity_set: List[str]
) -> AnnotatedDocument:
    annotations = set()
    offset = 0
    entities_pattern = [rf"<(.*?)>(.*?)</(.*?)>"]
    all_matches = [
        re.finditer(entity_pattern, annotated_text)
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
        annotated_text = annotated_text.replace(match.group(0), match.group(2))
    annotated_document = AnnotatedDocument(text=annotated_text, annotations=annotations)
    return annotated_document


def align_annotation(
    original_text: str, chatgpt_annotated_document: AnnotatedDocument
) -> AnnotatedDocument:
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

    for annotation in fixed_annotations_2.copy():
        if (
            (annotation.text not in original_text)
            | (annotation.start < 0)
            | (annotation.end < 0)
        ):
            fixed_annotations_2.remove(annotation)

    fixed_annotation.annotations = set(fixed_annotations_2)

    return fixed_annotation

def conll_to_inline_annotated_string(conll):
    inline_annotated_document = ""
    current_label = None
    for token, label in conll:
        if label == "O":
            if not current_label:
                inline_annotated_document += f"{token} "
            else:
                inline_annotated_document += f"</{current_label}> {token} "
                current_label = None
        else:
            current_label = label.split("-")[1]
            inline_annotated_document += f"<{current_label}>{token}"
    return inline_annotated_document.strip()


def annotated_document_to_conll(annotated_document: AnnotatedDocument) -> List[str]:
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


def annotated_document_to_inline_annotated_string(annotated_document):
    annotated_document = deepcopy(annotated_document)
    inline_annotated_string = annotated_document.text
    annotations = sorted(annotated_document.annotations, key=lambda x: x.start)
    for i in range(len(annotations)):
        annotation = annotations[i]
        start = annotation.start
        end = annotation.end
        label = annotation.label
        text = annotation.text
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
