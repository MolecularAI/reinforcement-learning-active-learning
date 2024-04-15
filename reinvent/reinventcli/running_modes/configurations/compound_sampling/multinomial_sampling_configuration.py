from pydantic import BaseModel


class MultinomialSamplingConfiguration(BaseModel):
    temperature: float = 1.0