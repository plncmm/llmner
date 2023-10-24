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
    annotated_document_to_few_shot_example
)


class BaseNer:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        max_tokens=256,
        stop=["###"],
    ):
        self.max_tokens = max_tokens
        self.stop = stop
        self.model = model

    def query_model(self, messages):
        chat = ChatOpenAI(
            model_name=self.model,
            max_tokens=self.max_tokens,
            model_kwargs={"presence_penalty": 0},
        )
        return chat(messages, stop=self.stop)


class ZeroShotNer(BaseNer):
    def contextualize(self, prompt_template, entities):
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

    def predict(self, x):
        messages = self.chat_template.format_messages(x=x)
        completion = self.query_model(messages)
        annotated_document = inline_annotation_to_annotated_document(
            completion.content, self.entities.keys()
        )
        aligned_annotated_document = align_annotation(x, annotated_document)
        y = aligned_annotated_document
        return y


class FewShotNer(ZeroShotNer):
    def contextualize(self, prompt_template, entities, examples):
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
