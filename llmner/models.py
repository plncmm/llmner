from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    AIMessagePromptTemplate,
)

from langchain.schema.messages import AIMessage

from langchain.chat_models import ChatOpenAI

from llmner.utils import (
    dict_to_enumeration,
    inline_annotation_to_annotated_document,
    inline_special_tokens_annotation_to_annotated_document,
    json_annotation_to_annotated_document,
    align_annotation,
    annotated_document_to_single_turn_few_shot_example,
    annotated_document_to_multi_turn_few_shot_example,
    detokenizer,
    annotated_document_to_conll,
    annotated_document_to_inline_annotated_string,
    annotated_document_to_json_annotated_string,
)

from llmner.templates import TEMPLATE_EN

from typing import List, Dict, Union, Tuple, Callable, Literal
from llmner.data import (
    AnnotatedDocument,
    AnnotatedDocumentWithException,
    NotContextualizedError,
    Conll,
    PromptTemplate,
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
        answer_shape: Literal["inline", "json"] = "inline",
        prompting_method: Literal["single_turn", "multi_turn"] = "single_turn",
        multi_turn_delimiters: Union[None, Tuple[str, str]] = None,
        augment_with_pos: Union[bool, Callable[[str], str]] = False,
        prompt_template: PromptTemplate = TEMPLATE_EN,
        system_message_as_user_message: bool = False,
    ):
        """NER model. Make sure you have at least the OPENAI_API_KEY environment variable set with your API key. Refer to the python openai library documentation for more information.

        Args:
            model (str, optional): Model name. Defaults to "gpt-3.5-turbo".
            max_tokens (int, optional): Max number of new tokens. Defaults to 256.
            stop (List[str], optional): List of strings that should stop generation. Defaults to ["###"].
            temperature (float, optional): Temperature for the generation. Defaults to 1.0.
            model_kwargs (Dict, optional): Arguments to pass to the llm. Defaults to {}. Refer to the OpenAI python library documentation and OpenAI API documentation for more information.
            answer_shape (Literal["inline", "json"], optional): Shape of the answer. The inline answer shape encloses entities between inline tags, as in '<LOC>Washington<LOC/>' and the json answer shapes expects a valid json response from the model. Defaults to "inline".
            prompting_method (Literal["single_turn", "multi_turn"], optional): Prompting method. In multi_turn, we query the model for each entity and at the end we compile the anotated document. Defaults to "single_turn".
            multi_turn_delimiters (Union[None, Tuple[str, str]], optional): Delimiter symbols for multi-turn prompting, the first element of the tuple is the start delimiter and the second element of the tuple is the end delimiter. Defaults to None.
            augment_with_pos (Union[bool, Callable[[str], str]], optional): If True, the model will be augmented with the part-of-speech tagging of the document. If a function is passed, the function will be called with the docuemnt as the argument and the returned value will be used as the augmentation. Defaults to False.
            prompt_template (str, optional): Prompt template to send the llm as the system message. Defaults to a prompt template for NER in English.
            system_message_as_user_message (bool, optional): If True, the system message will be sent as a user message. Defaults to False.
        """
        self.max_tokens = max_tokens
        self.stop = stop
        self.model = model
        self.chat_template = None
        self.model_kwargs = model_kwargs
        self.temperature = temperature
        self.answer_shape = answer_shape
        self.prompting_method = prompting_method
        self.multi_turn_delimiters = multi_turn_delimiters
        self.augment_with_pos = augment_with_pos
        self.prompt_template = prompt_template
        self.system_message_as_user_message = system_message_as_user_message

        self.multi_turn_prefix = self.prompt_template.multi_turn_prefix
        if self.multi_turn_delimiters:
            self.start_token = self.multi_turn_delimiters[0]
            self.end_token = self.multi_turn_delimiters[1]
        else:
            self.start_token = "###"
            self.end_token = "###"
        if (self.answer_shape == "inline") & (self.prompting_method == "single_turn"):
            current_prompt_template = self.prompt_template.inline_single_turn
        elif (self.answer_shape == "inline") & (self.prompting_method == "multi_turn"):
            if self.multi_turn_delimiters:
                current_prompt_template = (
                    self.prompt_template.inline_multi_turn_custom_delimiters
                )
            else:
                current_prompt_template = (
                    self.prompt_template.inline_multi_turn_default_delimiters
                )
        elif (self.answer_shape == "json") & (self.prompting_method == "single_turn"):
            current_prompt_template = self.prompt_template.json_single_turn
        elif (self.answer_shape == "json") & (self.prompting_method == "multi_turn"):
            current_prompt_template = self.prompt_template.json_multi_turn
        else:
            raise ValueError(
                "The answer shape and prompting method combination is not valid"
            )
        if not self.system_message_as_user_message:
            self.system_template = SystemMessagePromptTemplate.from_template(
                current_prompt_template
            )
        else:
            self.system_template = HumanMessagePromptTemplate.from_template(
                current_prompt_template
            )

    def query_model(
        self,
        messages: list,
        request_timeout: int = 600,
        remove_model_kwargs: bool = False,
    ):
        if remove_model_kwargs:
            model_kwargs = {}
        else:
            model_kwargs = self.model_kwargs
        chat = ChatOpenAI(
            model_name=self.model,  # type: ignore
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            model_kwargs=model_kwargs,
            request_timeout=request_timeout,
        )
        completion = chat.invoke(messages, stop=self.stop)
        return completion


class ZeroShotNer(BaseNer):
    """Zero-shot NER model class."""

    def contextualize(
        self,
        entities: Dict[str, str],
    ):
        """Method to ontextualize the zero-shot NER model. You don't need examples to contextualize this model.

        Args:
            entities (Dict[str, str]): Dict containing the entities to be recognized. The keys are the entity names and the values are the entity descriptions.
        """
        self.entities = entities
        if self.multi_turn_delimiters:
            self.system_message = self.system_template.format(
                entities=dict_to_enumeration(entities),
                entity_list=list(entities.keys()),
                start_token=self.start_token,
                end_token=self.end_token,
            )
        else:
            self.system_message = self.system_template.format(
                entities=dict_to_enumeration(entities),
                entity_list=list(entities.keys()),
            )
        if self.augment_with_pos:
            self.chat_template = ChatPromptTemplate.from_messages(
                [
                    self.system_message,
                    HumanMessagePromptTemplate.from_template(
                        f"{self.prompt_template.pos_answer_prefix} {{pos}}"
                    ),
                    HumanMessagePromptTemplate.from_template("{x}"),
                ]
            )
        else:
            self.chat_template = ChatPromptTemplate.from_messages(
                [
                    self.system_message,
                    HumanMessagePromptTemplate.from_template("{x}"),
                ]
            )

    def fit(self, *args, **kwargs):
        """Just a wrapper for the contextualize method. This method is here to be compatible with the sklearn API."""
        return self.contextualize(*args, **kwargs)

    def _predict_pos(self, x: str, request_timeout: int) -> str:
        pos_chat_template = ChatPromptTemplate.from_messages(
            [
                self.prompt_template.pos,
                HumanMessagePromptTemplate.from_template("{x}"),
            ]
        )
        messages = pos_chat_template.format_messages(x=x)
        completion = self.query_model(
            messages, request_timeout, remove_model_kwargs=True
        )
        return completion.content

    def _predict(
        self, x: str, request_timeout: int
    ) -> AnnotatedDocument | AnnotatedDocumentWithException:
        chat_template = self.chat_template
        if self.augment_with_pos:
            try:
                if callable(self.augment_with_pos):
                    pos = self.augment_with_pos(x)
                else:
                    pos = self._predict_pos(x, request_timeout)
            except Exception as e:
                logger.warning(
                    f"The pos completion for the text '{x}' raised an exception: {e}"
                )
                return AnnotatedDocumentWithException(
                    text=x, annotations=set(), exception=e
                )
            logger.debug(f"POS: {pos}")
            messages = chat_template.format_messages(x=x, pos=pos)
        else:
            messages = chat_template.format_messages(x=x)
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
        annotated_document = AnnotatedDocument(text=x, annotations=set())
        if self.answer_shape == "json":
            annotated_document = json_annotation_to_annotated_document(
                completion.content, list(self.entities.keys()), x
            )
        elif self.answer_shape == "inline":
            annotated_document = inline_annotation_to_annotated_document(
                completion.content, list(self.entities.keys())
            )

        aligned_annotated_document = align_annotation(x, annotated_document)
        y = aligned_annotated_document
        return y

    def _predict_multi_turn(
        self,
        x: str,
        request_timeout: int,
    ) -> AnnotatedDocument | AnnotatedDocumentWithException:
        chat_template = self.chat_template
        annotated_documents = []
        pos_added = False
        for entity in self.entities:
            human_msg_string = self.multi_turn_prefix + entity + ": " + x
            if bool(self.augment_with_pos) & (not pos_added):
                try:
                    if callable(self.augment_with_pos):
                        pos = self.augment_with_pos(x)
                    else:
                        pos = self._predict_pos(x, request_timeout)
                except Exception as e:
                    logger.warning(
                        f"The pos completion for the text '{x}' raised an exception: {e}"
                    )
                    return AnnotatedDocumentWithException(
                        text=x, annotations=set(), exception=e
                    )
                logger.debug(f"POS: {pos}")
                messages = chat_template.format_messages(
                    x=human_msg_string, pos=pos
                )
                pos_added = True
            else:
                messages = chat_template.format_messages(x=human_msg_string)

            try:
                completion = self.query_model(messages, request_timeout)
            except Exception as e:
                logger.warning(
                    f"The completion for the text '{x}' raised an exception: {e}"
                )
                return AnnotatedDocumentWithException(
                    text=x, annotations=set(), exception=e
                )
            logger.debug(
                f"Human message: {human_msg_string} \n Completion: {completion}"
            )

            annotated_document = AnnotatedDocument(text=x, annotations=set())
            if self.answer_shape == "json":
                annotated_document = json_annotation_to_annotated_document(
                    completion.content, list(self.entities.keys()), x
                )
            elif self.answer_shape == "inline":
                if self.multi_turn_delimiters:
                    annotated_document = (
                        inline_special_tokens_annotation_to_annotated_document(
                            completion.content, entity, self.start_token, self.end_token
                        )
                    )
                else:
                    annotated_document = inline_annotation_to_annotated_document(
                        completion.content, list(self.entities.keys())
                    )
            aligned_annotated_document = align_annotation(x, annotated_document)
            annotated_documents.append(aligned_annotated_document)
            if self.answer_shape == "inline":
                chat_template = ChatPromptTemplate.from_messages(
                    messages=messages
                    + [
                        AIMessage(
                            content=annotated_document_to_inline_annotated_string(
                                aligned_annotated_document,
                                custom_delimiters=self.multi_turn_delimiters,
                            )
                        ),
                        HumanMessagePromptTemplate.from_template("{x}"),
                    ]
                )
            elif self.answer_shape == "json":
                chat_template = ChatPromptTemplate.from_messages(
                    messages=messages
                    + [
                        AIMessage(
                            content=annotated_document_to_json_annotated_string(
                                aligned_annotated_document
                            )
                        ),
                        HumanMessagePromptTemplate.from_template("{x}"),
                    ]
                )
            else:
                raise ValueError("The answer shape is not valid")

        final_annotated_document = annotated_documents[0]
        for annotated_document in annotated_documents[1:]:
            final_annotated_document.annotations.update(annotated_document.annotations)

        return final_annotated_document

    def _predict_tokenized(self, x: List[str], request_timeout: int) -> Conll:
        detokenized_text = detokenizer(x)
        annotated_document = AnnotatedDocument(text=detokenized_text, annotations=set())
        if self.prompting_method == "single_turn":
            annotated_document = self._predict(detokenized_text, request_timeout)
        elif self.prompting_method == "multi_turn":
            annotated_document = self._predict_multi_turn(
                detokenized_text, request_timeout
            )
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
            if self.prompting_method == "single_turn":
                for annotated_document in tqdm(
                    executor.map(lambda x: self._predict(x, request_timeout), x),
                    disable=not progress_bar,
                    unit=" example",
                    total=len(x),
                ):
                    y.append(annotated_document)

            elif self.prompting_method == "multi_turn":
                for annotated_document in tqdm(
                    executor.map(
                        lambda x: self._predict_multi_turn(x, request_timeout), x
                    ),
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
            annotated_document = AnnotatedDocument(text=text, annotations=set())
            if self.prompting_method == "single_turn":
                annotated_document = self._predict(text, request_timeout)
            elif self.prompting_method == "multi_turn":
                annotated_document = self._predict_multi_turn(text, request_timeout)
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
        self, entities: Dict[str, str], examples: List[AnnotatedDocument]
    ):
        """Method to ontextualize the few-shot NER model. You need examples to contextualize this model.

        Args:
            entities (Dict[str, str]): Dict containing the entities to be recognized. The keys are the entity names and the values are the entity descriptions.
            examples (List[AnnotatedDocument]): List of AnnotatedDocument objects containing the annotated examples.
        """
        self.entities = entities
        if self.multi_turn_delimiters:
            self.system_message = self.system_template.format(
                entities=dict_to_enumeration(entities),
                entity_list=list(entities.keys()),
                start_token=self.start_token,
                end_token=self.end_token,
            )
        else:
            self.system_message = self.system_template.format(
                entities=dict_to_enumeration(entities),
                entity_list=list(entities.keys()),
            )
        example_template = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        if (self.answer_shape == "inline") & (self.prompting_method == "single_turn"):
            few_shot_template = FewShotChatMessagePromptTemplate(
                examples=list(
                    map(annotated_document_to_single_turn_few_shot_example, examples)
                ),
                example_prompt=example_template,
            )
        elif (self.answer_shape == "json") & (self.prompting_method == "single_turn"):
            few_shot_template = FewShotChatMessagePromptTemplate(
                examples=list(
                    map(
                        lambda x: annotated_document_to_single_turn_few_shot_example(
                            x, answer_shape="json"
                        ),
                        examples,
                    )
                ),
                example_prompt=example_template,
            )
        elif (self.answer_shape == "inline") & (self.prompting_method == "multi_turn"):
            few_shot_examples = []
            for example in examples:
                few_shot_examples.extend(
                    annotated_document_to_multi_turn_few_shot_example(
                        annotated_document=example,
                        multi_turn_prefix=self.multi_turn_prefix,
                        answer_shape="inline",
                        entity_set=list(self.entities.keys()),
                    )
                )
            few_shot_template = FewShotChatMessagePromptTemplate(
                examples=few_shot_examples,
                example_prompt=example_template,
            )
        elif (self.answer_shape == "json") & (self.prompting_method == "multi_turn"):
            few_shot_examples = []
            for example in examples:
                few_shot_examples.extend(
                    annotated_document_to_multi_turn_few_shot_example(
                        annotated_document=example,
                        multi_turn_prefix=self.multi_turn_prefix,
                        answer_shape="json",
                        entity_set=list(self.entities.keys()),
                    )
                )
            few_shot_template = FewShotChatMessagePromptTemplate(
                examples=few_shot_examples,
                example_prompt=example_template,
            )
        else:
            raise ValueError(
                "The answer shape and prompting method combination is not valid"
            )
        if self.augment_with_pos:
            raise NotImplementedError(
                "The augment_with_pos option is not implemented for the few-shot NER model"
            )
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                self.system_message,
                few_shot_template,
                HumanMessagePromptTemplate.from_template("{x}"),
            ]
        )
