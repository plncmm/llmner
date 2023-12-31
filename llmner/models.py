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
    detokenizer,
    annotated_document_to_conll,
)

from llmner.templates import SYSTEM_TEMPLATE_EN

from typing import List, Dict
from llmner.data import (
    AnnotatedDocument,
    AnnotatedDocumentWithException,
    NotContextualizedError,
    Conll,
)

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()

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
        temperature: float = 1.0,
        model_kwargs: Dict = {},
    ):
        """NER model. Make sure you have at least the OPENAI_API_KEY environment variable set with your API key. Refer to the python openai library documentation for more information.

        Args:
            model (str, optional): Model name. Defaults to "gpt-3.5-turbo".
            max_tokens (int, optional): Max number of new tokens. Defaults to 256.
            stop (List[str], optional): List of strings that should stop generation. Defaults to ["###"].
            temperature (float, optional): Temperature for the generation. Defaults to 1.0.
            model_kwargs (Dict, optional): Arguments to pass to the llm. Defaults to {}. Refer to the OpenAI python library documentation and OpenAI API documentation for more information.
        """
        self.max_tokens = max_tokens
        self.stop = stop
        self.model = model
        self.chat_template = None
        self.model_kwargs = model_kwargs
        self.temperature = temperature

    def query_model(self, messages: list, request_timeout: int = 600):
        chat = ChatOpenAI(
            model_name=self.model,  # type: ignore
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            model_kwargs=self.model_kwargs,
            request_timeout=request_timeout,
        )
        completion = chat.invoke(messages, stop=self.stop)
        return completion


class ZeroShotNer(BaseNer):
    """Zero-shot NER model class."""

    def contextualize(
        self,
        entities: Dict[str, str],
        prompt_template: str = SYSTEM_TEMPLATE_EN,
        system_message_as_user_message: bool = False,
    ):
        """Method to ontextualize the zero-shot NER model. You don't need examples to contextualize this model.

        Args:
            entities (Dict[str, str]): Dict containing the entities to be recognized. The keys are the entity names and the values are the entity descriptions.
            prompt_template (str, optional): Prompt template to send the llm as the system message. Defaults to a prompt template for NER in English.
            system_message_as_user_message (bool, optional): If True, the system message will be sent as a user message. Defaults to False.
        """
        self.entities = entities
        if not system_message_as_user_message:
            system_template = SystemMessagePromptTemplate.from_template(prompt_template)
        else:
            system_template = HumanMessagePromptTemplate.from_template(prompt_template)
        self.system_message = system_template.format(
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

    def _predict(
        self, x: str, request_timeout: int
    ) -> AnnotatedDocument | AnnotatedDocumentWithException:
        messages = self.chat_template.format_messages(x=x)
        try:
            completion = self.query_model(messages, request_timeout)
        except Exception as e:
            logger.warning(
                f"The completion for the text '{x}' raised an exception: {e}"
            )
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

    def _predict_tokenized(self, x: List[str], request_timeout: int) -> Conll:
        detokenized_text = detokenizer(x)
        annotated_document = self._predict(detokenized_text, request_timeout)
        if isinstance(annotated_document, AnnotatedDocumentWithException):
            logger.warning(
                f"The completion for the text '{detokenized_text}' raised an exception: {annotated_document.exception}"
            )
        conll = annotated_document_to_conll(annotated_document)
        if not len(x) == len(conll):
            logger.warning(
                "The number of tokens and the number of conll tokens are different"
            )
        return conll

    def _predict_parallel(
        self, x: List[str], max_workers: int, progress_bar: bool, request_timeout: int
    ) -> List[AnnotatedDocument | AnnotatedDocumentWithException]:
        y = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for annotated_document in tqdm(
                executor.map(lambda x: self._predict(x, request_timeout), x),
                disable=not progress_bar,
                unit=" example",
                total=len(x),
            ):
                y.append(annotated_document)
        return y

    def _predict_tokenized_parallel(
        self,
        x: List[List[str]],
        max_workers: int,
        progress_bar: bool,
        request_timeout: int,
    ) -> List[List[Conll]]:
        y = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for conll in tqdm(
                executor.map(lambda x: self._predict_tokenized(x, request_timeout), x),
                disable=not progress_bar,
                unit=" example",
                total=len(x),
            ):
                y.append(conll)
        return y

    def _predict_serial(
        self, x: List[str], progress_bar: bool, request_timeout: int
    ) -> List[AnnotatedDocument | AnnotatedDocumentWithException]:
        y = []
        for text in tqdm(x, disable=not progress_bar, unit=" example"):
            annotated_document = self._predict(text, request_timeout)
            y.append(annotated_document)
        return y

    def _predict_tokenized_serial(
        self, x: List[List[str]], progress_bar: bool, request_timeout: int
    ) -> List[List[Conll]]:
        y = []
        for tokenized_text in tqdm(x, disable=not progress_bar, unit=" example"):
            conll = self._predict_tokenized(tokenized_text, request_timeout)
            y.append(conll)
        return y

    def predict(
        self,
        x: List[str],
        progress_bar: bool = True,
        max_workers: int = 1,
        request_timeout: int = 600,
    ) -> List[AnnotatedDocument | AnnotatedDocumentWithException]:
        """Method to perform NER on a list of strings.

        Args:
            x (List[str]): List of strings.
            progress_bar (bool, optional): If True, a progress bar will be displayed. Defaults to True.
            max_workers (int, optional): Number of workers to use for parallel processing. If -1, the number of workers will be equal to the number of CPU cores. Defaults to 1.
            request_timeout (int, optional): Timeout in seconds for the requests. Defaults to 600 seconds.

        Raises:
            NotContextualizedError: Error if the model is not contextualized before calling the predict method.
            ValueError: The input must be a list of strings.

        Returns:
            List[AnnotatedDocument | AnnotatedDocumentWithException]: List of AnnotatedDocument objects if there were no exceptions, a list of AnnotatedDocumentWithException objects if there were exceptions.
        """
        if self.chat_template is None:
            raise NotContextualizedError(
                "You must call the contextualize method before calling the predict method"
            )
        if not isinstance(x, list):
            raise ValueError("x must be a list")
        if isinstance(x[0], str):
            if max_workers == -1:
                y = self._predict_parallel(x, CPU_COUNT, progress_bar, request_timeout)
            elif max_workers == 1:
                y = self._predict_serial(x, progress_bar, request_timeout)
            elif max_workers > 1:
                y = self._predict_parallel(
                    x, max_workers, progress_bar, request_timeout
                )
            else:
                raise ValueError("max_workers must be greater than 0")
        else:
            raise ValueError(
                "x must be a list of strings, maybe you want to use predict_tokenized instead?"
            )
        return y

    def predict_tokenized(
        self,
        x: List[List[str]],
        progress_bar: bool = True,
        max_workers: int = 1,
        request_timeout: int = 600,
    ) -> List[List[Conll]]:
        """Method to perform NER on a list of tokenized documents.

        Args:
            x (List[List[str]]): List of lists of tokens.
            progress_bar (bool, optional): If True, a progress bar will be displayed. Defaults to True.
            max_workers (int, optional): Number of workers to use for parallel processing. If -1, the number of workers will be equal to the number of CPU cores. Defaults to 1.
            request_timeout (int, optional): Timeout in seconds for the requests. Defaults to 600 seconds.

        Returns:
            List[List[Conll]]: List of lists of tuples of (token, label).
        """
        if not isinstance(x, list):
            raise ValueError("x must be a list")
        if isinstance(x[0], list):
            if max_workers == -1:
                y = self._predict_tokenized_parallel(
                    x, CPU_COUNT, progress_bar, request_timeout
                )
            elif max_workers == 1:
                y = self._predict_tokenized_serial(x, progress_bar, request_timeout)
            elif max_workers > 1:
                y = self._predict_tokenized_parallel(
                    x, max_workers, progress_bar, request_timeout
                )
            else:
                raise ValueError("max_workers must be greater than 0")
        else:
            raise ValueError(
                "x must be a list of lists of tokens, maybe you want to use predict instead?"
            )
        return y


class FewShotNer(ZeroShotNer):
    def contextualize(
        self,
        entities: Dict[str, str],
        examples: List[AnnotatedDocument],
        prompt_template: str = SYSTEM_TEMPLATE_EN,
        system_message_as_user_message: bool = False,
    ):
        """Method to ontextualize the few-shot NER model. You need examples to contextualize this model.

        Args:
            entities (Dict[str, str]): Dict containing the entities to be recognized. The keys are the entity names and the values are the entity descriptions.
            examples (List[AnnotatedDocument]): List of AnnotatedDocument objects containing the annotated examples.
            prompt_template (str, optional): Prompt template to send the llm as the system message. Defaults to a prompt template for NER in English. Defaults to a prompt template for NER in English.
            system_message_as_user_message (bool, optional): If True, the system message will be sent as a user message. Defaults to False.
        """
        self.entities = entities
        if not system_message_as_user_message:
            system_template = SystemMessagePromptTemplate.from_template(prompt_template)
        else:
            system_template = HumanMessagePromptTemplate.from_template(prompt_template)
        self.system_message = system_template.format(
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
