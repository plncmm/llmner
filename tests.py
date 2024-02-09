import unittest
from llmner import ZeroShotNer, FewShotNer
from llmner.data import AnnotatedDocument, Annotation
from llmner.utils import conll_to_annotated_document
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

entities = {
    "person": "A person name, it can include first and last names, for example: Fabián Villena, Claudio or Luis Miranda",
    "organization": "An organization name, it can be a company, a government agency, etc.",
    "location": "A location name, it can be a city, a country, etc.",
}

examples = [
    AnnotatedDocument(
        text="Gabriel Boric is the president of Chile",
        annotations={
            Annotation(start=34, end=39, label="location"),
            Annotation(start=0, end=13, label="person"),
        },
    ),
    AnnotatedDocument(
        text="Elon Musk is the owner of the US company Tesla",
        annotations={
            Annotation(start=30, end=32, label="location"),
            Annotation(start=0, end=9, label="person"),
            Annotation(start=41, end=46, label="organization"),
        },
    ),
    AnnotatedDocument(
        text="Bill Gates is the owner of Microsoft",
        annotations={
            Annotation(start=0, end=10, label="person"),
            Annotation(start=27, end=36, label="organization"),
        },
    ),
    AnnotatedDocument(
        text="John is the president of Argentina and he visited Chile last week",
        annotations={
            Annotation(start=0, end=4, label="person"),
            Annotation(start=25, end=34, label="location"),
            Annotation(start=50, end=55, label="location"),
        },
    ),
]

x = [
    "Pedro Pereira is the president of Perú and the owner of Walmart.",
    "John Kennedy was the president of the United States of America.",
    "Jeff Bezos is the owner of Amazon.",
    "Jocelyn Dunstan is a female scientist from Chile",
]

x_tokenized = [
    [
        "Pedro",
        "Pereira",
        "is",
        "the",
        "president",
        "of",
        "Perú",
        "and",
        "the",
        "owner",
        "of",
        "Walmart",
        ".",
    ],
    [
        "John",
        "Kennedy",
        "was",
        "the",
        "president",
        "of",
        "the",
        "United",
        "States",
        "of",
        "America",
        ".",
    ],
    ["Jeff", "Bezos", "is", "the", "owner", "of", "Amazon", "."],
    ["Jocelyn", "Dunstan", "is", "a", "female", "scientist", "from", "Chile"],
]

y = [
    AnnotatedDocument(
        text="Pedro Pereira is the president of Perú and the owner of Walmart.",
        annotations={
            Annotation(start=34, end=38, label="location", text="Perú"),
            Annotation(start=0, end=13, label="person", text="Pedro Pereira"),
            Annotation(start=56, end=63, label="organization", text="Walmart"),
        },
    ),
    AnnotatedDocument(
        text="John Kennedy was the president of the United States of America.",
        annotations={
            Annotation(
                start=38, end=62, label="location", text="United States of America"
            ),
            Annotation(start=0, end=12, label="person", text="John Kennedy"),
        },
    ),
    AnnotatedDocument(
        text="Jeff Bezos is the owner of Amazon.",
        annotations={
            Annotation(start=0, end=10, label="person", text="Jeff Bezos"),
            Annotation(start=27, end=33, label="organization", text="Amazon"),
        },
    ),
    AnnotatedDocument(
        text="Jocelyn Dunstan is a female scientist from Chile",
        annotations={
            Annotation(start=43, end=48, label="location", text="Chile"),
            Annotation(start=0, end=15, label="person", text="Jocelyn Dunstan"),
        },
    ),
]

y_conll = [
    [
        ("Pedro", "B-person"),
        ("Pereira", "I-person"),
        ("is", "O"),
        ("the", "O"),
        ("president", "O"),
        ("of", "O"),
        ("Perú", "B-location"),
        ("and", "O"),
        ("the", "O"),
        ("owner", "O"),
        ("of", "O"),
        ("Walmart", "B-organization"),
        (".", "O"),
    ],
    [
        ("John", "B-person"),
        ("Kennedy", "I-person"),
        ("was", "O"),
        ("the", "O"),
        ("president", "O"),
        ("of", "O"),
        ("the", "O"),
        ("United", "B-location"),
        ("States", "I-location"),
        ("of", "I-location"),
        ("America", "I-location"),
        (".", "O"),
    ],
    [
        ("Jeff", "B-person"),
        ("Bezos", "I-person"),
        ("is", "O"),
        ("the", "O"),
        ("owner", "O"),
        ("of", "O"),
        ("Amazon", "B-organization"),
        (".", "O"),
    ],
    [
        ("Jocelyn", "B-person"),
        ("Dunstan", "I-person"),
        ("is", "O"),
        ("a", "O"),
        ("female", "O"),
        ("scientist", "O"),
        ("from", "O"),
        ("Chile", "B-location"),
    ],
]


def iou(annotations_true, annotations_predicted) -> float:
    # intersection over union
    intersection = annotations_true.intersection(annotations_predicted)
    union = annotations_true.union(annotations_predicted)
    return len(intersection) / len(union)


def assert_equal_annotated_documents(
    annotated_documents_true,
    annotated_documents,
    iou_threshold: float = 1.0,
    tokenized: bool = False,
):
    if tokenized:
        annotated_documents_true = [
            conll_to_annotated_document(doc) for doc in annotated_documents_true
        ]
        annotated_documents = [
            conll_to_annotated_document(doc) for doc in annotated_documents
        ]
    close_enough = False
    annotations_not_equal = []
    for annotated_document_true, annotated_document in zip(
        annotated_documents_true, annotated_documents
    ):
        iou_value = iou(
            annotated_document_true.annotations, annotated_document.annotations
        )
        if iou_value == 0:
            raise AssertionError(
                f"Annotations are not equal. Expected: {annotated_document_true.annotations}, got: {annotated_document.annotations}"
            )
        elif iou_value >= iou_threshold:
            close_enough = True
        else:
            annotations_not_equal.append(
                (
                    annotated_document_true.annotations,
                    annotated_document.annotations,
                    iou_value,
                )
            )

    error = ""
    for annotations_true, annotations_predicted, iou_value in annotations_not_equal:
        error += f"\nExpected: {annotations_true}, got: {annotations_predicted}, iou: {iou_value}"
    if not close_enough:
        raise AssertionError(f"Annotations are not equal. {error}")
    elif len(annotations_not_equal) > 0:
        logger.warning(f"Annotations are not perfectly equal. {error}")

    return True


def test_model(
    few_shot: bool,
    model_kwargs: dict,
    contextualize_kwargs: dict,
    iou_threshold: float = 1.0,
    tokenized: bool = False,
) -> bool:
    if not few_shot:
        model = ZeroShotNer(**model_kwargs)
        model.contextualize(**contextualize_kwargs)
    else:
        model = FewShotNer(**model_kwargs)
        model.contextualize(**contextualize_kwargs)
    if not tokenized:
        annotated_documents = model.predict(x, max_workers=1)
        assert_equal_annotated_documents(
            y, annotated_documents, iou_threshold=iou_threshold
        )
    if tokenized:
        annotated_documents_conll = model.predict_tokenized(x_tokenized, max_workers=-1)
        assert_equal_annotated_documents(
            y_conll,
            annotated_documents_conll,
            iou_threshold=iou_threshold,
            tokenized=True,
        )
    return True


class TestZeroShotNer(unittest.TestCase):
    # Single-turn test cases

    def test_zero_shot_inline_single_turn_posfalse(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="single_turn",
                multi_turn_delimiters=None,
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities),
        )

    def test_zero_shot_json_single_turn_posfalse(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="single_turn",
                multi_turn_delimiters=None,
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities),
        )

    def test_zero_shot_inline_single_turn_postrue(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="single_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities),
        )

    def test_zero_shot_json_single_turn_postrue(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="single_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities),
        )

    # Multi-turn test cases

    def test_zero_shot_inline_multi_turn_default_delimiters_posfalse(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities),
            iou_threshold=1,
        )

    def test_zero_shot_inline_multi_turn_custom_delimiters_posfalse(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=("@", "@"),
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities),
            iou_threshold=1.0,
        )

    def test_zero_shot_inline_multi_turn_default_delimiters_postrue(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities),
            iou_threshold=1.0,
        )

    def test_zero_shot_inline_multi_turn_custom_delimiters_postrue(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=("@", "@"),
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities),
            iou_threshold=1.0,
        )

    def test_zero_shot_json_multi_turn_posfalse(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities),
            iou_threshold=1.0,
        )

    def test_zero_shot_json_multi_turn_postrue(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities),
            iou_threshold=1.0,
        )
    
    def test_zero_shot_json_multi_turn_postrue_final_message(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
                final_message_with_all_entities=True,
            ),
            contextualize_kwargs=dict(entities=entities),
            iou_threshold=1.0,
        )


class TestFewShotNer(unittest.TestCase):
    # Single-turn test cases

    def test_few_shot_inline_single_turn_posfalse(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="single_turn",
                multi_turn_delimiters=None,
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
        )

    def test_few_shot_json_single_turn_posfalse(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="single_turn",
                multi_turn_delimiters=None,
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
        )

    def test_few_shot_inline_single_turn_postrue(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="single_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
        )

    def test_few_shot_json_single_turn_postrue(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="single_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
        )

    # multi-turn test cases

    def test_few_shot_inline_multi_turn_default_delimiters_posfalse(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
        )

    def test_few_shot_inline_multi_turn_custom_delimiters_posfalse(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=("@", "@"),
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
        )

    def test_few_shot_inline_multi_turn_default_delimiters_postrue(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
        )

    def test_few_shot_inline_multi_turn_custom_delimiters_postrue(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=("@", "@"),
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
        )

    def test_few_shot_json_multi_turn_posfalse(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
        )

    def test_few_shot_json_multi_turn_postrue(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
        )
    
    def test_few_shot_json_multi_turn_postrue_final_message(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
                final_message_with_all_entities=True,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
        )


class TestPredictTokenized(unittest.TestCase):
    def test_zero_shot_inline_single_turn_posfalse_tokenized(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="single_turn",
                multi_turn_delimiters=None,
                augment_with_pos=False,
            ),
            contextualize_kwargs=dict(entities=entities),
            tokenized=True,
        )

    def test_few_shot_json_multi_turn_postrue_tokenized(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="json",
                prompting_method="multi_turn",
                multi_turn_delimiters=None,
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
            tokenized=True,
        )


class TestCustomDelimiters(unittest.TestCase):
    def test_zero_shot_inline_multi_turn_custom_delimiters_postrue_doubleat(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=("@@", "@@"),
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities),
            iou_threshold=1.0,
        )

    def test_few_shot_inline_multi_turn_custom_delimiters_postrue_doubleat(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=("@@", "@@"),
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
        )

    def test_zero_shot_inline_multi_turn_custom_delimiters_postrue_assymetric(self):
        test_model(
            few_shot=False,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=("@", "#"),
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities),
            iou_threshold=1.0,
        )

    def test_few_shot_inline_multi_turn_custom_delimiters_postrue_assymetric(self):
        test_model(
            few_shot=True,
            model_kwargs=dict(
                answer_shape="inline",
                prompting_method="multi_turn",
                multi_turn_delimiters=("@", "#"),
                augment_with_pos=True,
            ),
            contextualize_kwargs=dict(entities=entities, examples=examples),
            iou_threshold=1.0,
        )
