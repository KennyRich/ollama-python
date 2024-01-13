from pydantic import BaseModel, Field


class Embedding(BaseModel):
    """A model embedding"""

    embedding: list[float] = Field(..., description="The embedding of the text")
