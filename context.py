from pydantic import BaseModel, Field
from typing import Dict, Any

class ModelContext(BaseModel):
    llm_model: str = Field(default="qwen3:1.7b")
    embedding_model: str = Field(default="nomic-embed-text:latest")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    collection: str = Field(default="default")
    namespace: str = Field(default="global")
    temperature: float = Field(default=0.3)
    k: int = Field(default=5)
    metadata: Dict[str, Any] = Field(default_factory=dict)
