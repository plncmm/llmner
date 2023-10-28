from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI

from llmner.utils import (
    dict_to_enumeration,
    inline_annotation_to_annotated_document,
    align_annotation,
    annotated_document_to_few_shot_example,
)

from templates import SYSTEM_TEMPLATE_EN

from typing import List, Dict
from llmner.data import AnnotatedDocument, AnnotatedDocumentWithException, NotContextualizedError

import logging

logger = logging.getLogger(__name__)


class BaseNer:
    """Base NER model class. All NER models should inherit from this class."""

    def __init__(
        # TODO: add env variables
        self,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 256,
        stop: List[str] = ["###"],
        model_kwargs: Dict = {},
    ):
        """NER model. Make sure you have at least the OPENAI_API_KEY environment variable set with your API key. Refer to the python openai library documentation for more information.

        Args:
            model (str, optional): Model name. Defaults to "gpt-3.5-turbo".
            max_tokens (int, optional): Max number of new tokens. Defaults to 256.
            stop (List[str], optional): List of strings that should stop generation. Defaults to ["###"].
            model_kwargs (Dict, optional): Arguments to pass to the llm. Defaults to {}. Refer to the OpenAI python library documentation and OpenAI API documentation for more information.
        """
        self.max_tokens = max_tokens
        self.stop = stop
        self.model = model
        self.chat_template = None
        self.model_kwargs = model_kwargs

    def query_model(self, messages: list):
        chat = ChatOpenAI(
            model_name=self.model,  # type: ignore
            max_tokens=self.max_tokens,
            model_kwargs=self.model_kwargs,
        )
        return chat(messages, stop=self.stop)


class ZeroShotNer(BaseNer):
    """Zero-shot NER model class."""

    def contextualize(
        self, entities: Dict[str, str], prompt_template: str = SYSTEM_TEMPLATE_EN
    ):
        """Method to ontextualize the zero-shot NER model. You don't need examples to contextualize this model.

        Args:
            entities (Dict[str, str]): Dict containing the entities to be recognized. The keys are the entity names and the values are the entity descriptions.
            prompt_template (str, optional): Prompt template to send the llm as the system message. Defaults to a prompt template for NER in English.
        """
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
        """Just a wrapper for the contextualize method. This method is here to be compatible with the sklearn API."""
        return self.contextualize(*args, **kwargs)

    def _predict(self, x: str) -> AnnotatedDocument | AnnotatedDocumentWithException:
        messages = self.chat_template.format_messages(x=x)
        try:
            completion = self.query_model(messages)
        except Exception as e:
            logger.warning(f"The completion for the text '{x}' raised an exception: {e}")
            return AnnotatedDocumentWithException(
                text=x, annotations=set(), exception=e
            )
        logger.debug(f"Completion: {completion}")
        annotated_document = inline_annotation_to_annotated_document(
            completion.content, list(self.entities.keys())
        )
        aligned_annotated_document = align_annotation(x, annotated_document)
        y = aligned_annotated_document
        return y

    def predict(self, x: List[str]) -> List[AnnotatedDocument]:
        """Method to perform NER on a list of strings.

        Args:
            x (List[str]): List of strings to perform NER on.

        Raises:
            NotContextualizedError: Error if the model is not contextualized before calling the predict method.
            ValueError: The input must be a list of strings.

        Returns:
            List[AnnotatedDocument]: List of AnnotatedDocument objects.
        """
        if self.chat_template is None:
            raise NotContextualizedError(
                "You must call the contextualize method before calling the predict method"
            )
        if not isinstance(x, list):
            raise ValueError("x must be a list of strings")
        return list(map(self._predict, x))


class FewShotNer(ZeroShotNer):
    def contextualize(
        self,
        entities: Dict[str, str],
        examples: List[AnnotatedDocument],
        prompt_template: str = SYSTEM_TEMPLATE_EN,
    ):
        """Method to ontextualize the few-shot NER model. You need examples to contextualize this model.

        Args:
            entities (Dict[str, str]): Dict containing the entities to be recognized. The keys are the entity names and the values are the entity descriptions.
            examples (List[AnnotatedDocument]): List of AnnotatedDocument objects containing the annotated examples.
            prompt_template (str, optional): Prompt template to send the llm as the system message. Defaults to a prompt template for NER in English. Defaults to a prompt template for NER in English.
        """
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
