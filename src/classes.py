from pydantic import BaseModel


class Typeinfo(BaseModel):
    type: str

class Function(BaseModel):
    name: str
    description: str
    parameters: dict[str, Typeinfo]
    returns: Typeinfo

class PromptTest(BaseModel):
    prompt: str

class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: str

class TokenSets(BaseModel):
    """Token ids grouped by what they are allowed to be.
 
    Built once from the vocabulary, then reused for every prompt.
    """
 
    number_ids: list[int]
    string_ids: list[int]
    quote_ids: list[int]
    stop_ids: list[int]