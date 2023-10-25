from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI

from utils import (
    dict_to_enumeration,
    inline_annotation_to_annotated_document,
    align_annotation,
    annotated_document_to_few_shot_example,
)

from templates import SYSTEM_TEMPLATE_EN

from typing import List, Dict
from data import AnnotatedDocument, NotContextualizedError

import logging

logger = logging.getLogger(__name__)


class BaseNer:
    def __init__(
        # TODO: add env variables
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 256,
        stop: List[str] = ["###"],
    ):
        self.max_tokens = max_tokens
        self.stop = stop
        self.model = model
        self.chat_template = None

    def query_model(self, messages: list):
        chat = ChatOpenAI(
            model_name=self.model,  # type: ignore
            max_tokens=self.max_tokens,
            model_kwargs={"presence_penalty": 0},
        )
        return chat(messages, stop=self.stop)


class ZeroShotNer(BaseNer):
    def contextualize(
        self, entities: Dict[str, str], prompt_template: str = SYSTEM_TEMPLATE_EN
    ):
        self.entities = entities
        self.system_message = SystemMessagePromptTemplate.from_template(
            prompt_template
        ).format(
            entities=dict_to_enumeration(entities), entity_list=list(entities.keys())
        )
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                self.system_message,
                HumanMessagePromptTemplate.from_template("{x}"),
            ]
        )

    def fit(self, *args, **kwargs):
        return self.contextualize(*args, **kwargs)

    def _predict(self, x: str) -> AnnotatedDocument:
        messages = self.chat_template.format_messages(x=x)
        completion = self.query_model(messages)
        logger.debug(f"Completion: {completion}")
        annotated_document = inline_annotation_to_annotated_document(
            completion.content, list(self.entities.keys())
        )
        aligned_annotated_document = align_annotation(x, annotated_document)
        y = aligned_annotated_document
        return y

    def predict(self, x: List[str]) -> List[AnnotatedDocument]:
        if self.chat_template is None:
            raise NotContextualizedError(
                "You must call the contextualize method before calling the predict method"
            )
        return list(map(self._predict, x))


class FewShotNer(ZeroShotNer):
    def contextualize(
        self,
        entities: Dict[str, str],
        examples: List[AnnotatedDocument],
        prompt_template: str = SYSTEM_TEMPLATE_EN,
    ):
        self.entities = entities
        self.system_message = SystemMessagePromptTemplate.from_template(
            prompt_template
        ).format(
            entities=dict_to_enumeration(entities), entity_list=list(entities.keys())
        )
        example_template = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        few_shot_template = FewShotChatMessagePromptTemplate(
            examples=list(map(annotated_document_to_few_shot_example, examples)),
            example_prompt=example_template,
        )
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                self.system_message,
                few_shot_template,
                HumanMessagePromptTemplate.from_template("{x}"),
            ]
        )
