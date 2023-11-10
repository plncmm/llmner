# llmNER: (Few|Zero)-Shot Named Entity Recognition without training data

Exploit the power of Large Language Models (LLM) to perform Zero-Shot or Few-Shot Named Entity Recognition (NER) without the need of annotated data.

## Installation

```bash
pip install git+https://github.com/plncmm/llmner.git
```

## Usage

### Zero-Shot NER

```python
import os
os.environ["OPENAI_API_KEY"] = "<your OpenAI api key>"
from llmner import ZeroShotNer

entities = {
    "person": "A person name, it can include first and last names, for example: John Kennedy and Bill Gates",
    "organization": "An organization name, it can be a company, a government agency, etc.",
    "location": "A location name, it can be a city, a country, etc.",
}

model = ZeroShotNer()
model.contextualize(entities=entities)

model.predict(["Pedro Pereira is the president of Perú and the owner of Walmart."])
```

### Few-Shot NER

```python
import os
os.environ["OPENAI_API_KEY"] = "<your OpenAI api key>"
from llmner import FewShotNer
from llmner.data import AnnotatedDocument, Annotation

entities = {
    "person": "A person name, it can include first and last names, for example: John Kennedy and Bill Gates",
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
        text="John is the president of Argentina",
        annotations={
            Annotation(start=0, end=4, label="person"),
            Annotation(start=25, end=34, label="location"),
        },
    ),
]

model = FewShotNer()
model.contextualize(entities=entities, examples=examples)

model.predict(["Pedro Pereira is the president of Perú and the owner of Walmart."])
```

### Use your own LLM

You need to set the next environment variables to use your own LLM:

- `OPENAI_API_KEY`: Your API key if you need one, otherwise use a random one.
- `OPENAI_API_BASE`: The API base URL

### If you belong to an OpenAI organization

You need to set the next environment variables to use your organization:

- `OPENAI_ORG_ID`: Your organization ID
- `OPENAI_API_KEY`: Your OpenAi API key

### If you are using the model through Azure

You need to set the next environment variables to use your LLM throgh Azure:

- `OPENAI_API_BASE`: The Azure API base URL
- `OPENAI_API_KEY`: Your Azure API key
- `OPENAI_API_TYPE`: You need to set it to `azure`
- `OPENAI_API_VERSION`: You need to set it to your Azure API version

Also, when instantiating the model object you need to pass `model_kwargs={"engine":"<your engine name>"}`

For example:

```python
import os
os.environ["OPENAI_API_KEY"] = "<your API key>"
os.environ["OPENAI_API_BASE"] = "<your Azure API base URL>"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "<your Azure API version>"
from llmner import ZeroShotNer

entities = {
    "person": "A person name, it can include first and last names, for example: John Kennedy and Bill Gates",
    "organization": "An organization name, it can be a company, a government agency, etc.",
    "location": "A location name, it can be a city, a country, etc.",
}

model = ZeroShotNer(model_kwargs={"engine":"<your engine name>"})
model.contextualize(entities=entities)

model.predict(["Pedro Pereira is the president of Perú and the owner of Walmart."])
```

## If you are using Deep Infra

You have to set `OPENAI_API_BASE` to `https://api.deepinfra.com/v1/openai`, and `OPENAI_API_TYPE` to your API key and instantiate the models setting the `model` argument to the name of the deployed model. For example if you want perform Few-Shot NER using Llama 2 70B you need to instantiate the model as follows: `ZeroShotNer(model="meta-llama/Llama-2-70b-chat-hf")`.