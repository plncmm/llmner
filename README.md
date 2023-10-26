# LLMNER: Named Entity Recognition without training data

Exploit the power of Large Language Models (LLM) to perform Named Entity Recognition (NER) without the need of annotated data.

## Installation

```bash
pip install llmner
```

## Usage

```python
import os
os.environ["OPENAI_API_KEY"] = "<your api key>"
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


