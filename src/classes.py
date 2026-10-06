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
