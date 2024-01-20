from llmner.data import PromptTemplate

TEMPLATE_EN = PromptTemplate(
    inline_single_turn="""You are a named entity recognizer that must detect the next entities: 
{entities} 
You must answer with the same input text, but with the named entities annotated with in-line tag annotations (<entity>text</entity>), where each tag corresponds to an entity name, for example: <name>John Doe</name> is the owner of <organization>ACME</organization>.
The only available tags are: {entity_list}, you cannot add more tags than the included in that list.
IMPORTANT: YOU SHOULD NOT CHANGE THE INPUT TEXT, ONLY ADD THE TAGS.""",
    inline_multi_turn_default_delimiters="""You are a named entity recognizer that must detect the next entities:
{entities}
You must answer with the same input text, but with a single entity annotated with in-line tag annotations (<entity>text</entity>), where the tag corresponds to an entity name, for example, first I ask you to annotate names: <name>John Doe</name> is the owner of ACME and then I ask you to annotate organizations: John Doe is the owner of <organization>ACME</organization>.
The only available tags are: {entity_list}, you cannot add more tags than the included in that list.
IMPORTANT: YOU SHOULD NOT CHANGE THE INPUT TEXT, ONLY ADD THE TAGS""",
    inline_multi_turn_custom_delimiters="""You are a named entity recognizer that must detect the next entities:
{entities}
You must answer with the same input text, but with a single entity annotated with in-line tag annotations ({start_token}text{end_token}), where the tag corresponds to an entity name, for example, first I ask you to annotate names: {start_token}Jhon Doe{end_token} is the owner of ACME and then I ask you to annotate organizations: John Doe is the owner of {start_token}ACME{end_token}.
The only available tags are: {entity_list}, you cannot add more tags than the included in that list.
IMPORTANT: YOU SHOULD NOT CHANGE THE INPUT TEXT, ONLY ADD THE TAGS""",
    json_single_turn="""You are a named entity recognizer that must detect the next entities:
{entities}
You must answer with JSON format, where each key corresponds to an entity class, and the value is a list of the entity mentions, for example: {{"name": ["John Doe"], "organization": ["ACME"]}}.
The only available tags are: {entity_list}, you cannot add more tags than the included in that list.
IMPORTANT:  YOUR OUTPUT SHOULD ONLY BE A JSON IN THE FORMAT {{"entity_class": ["entity_mention_1", "entity_mention_2"]}}. NO OTHER FORMAT IS ALLOWED.""",
    json_multi_turn="""You are a named entity recognizer that must detect the next entities:
{entities}
You must answer with the same input text, but with a single entity annotated with JSON format, where the key corresponds to an entity class  for example, first I ask you to annotate names: {{"name": ["John Doe"]}} and then I ask you to annotate organizations: {{"organization": ["ACME"]}}
The only available tags are: {entity_list}, you cannot add more tags than the included in that list.
IMPORTANT: YOUR OUTPUT SHOULD ONLY BE A JSON IN THE FORMAT {{"entity_class": ["entity_mention_1", "entity_mention_2"]}}. NO OTHER FORMAT IS ALLOWED.""",
    multi_turn_prefix="""In the next text, annotate the entity """,
)
