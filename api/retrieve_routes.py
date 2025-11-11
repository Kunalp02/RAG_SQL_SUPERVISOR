from fastapi import APIRouter
from pydantic import BaseModel
from context import ModelContext
from helpers import get_vectorstore

router = APIRouter(prefix="/retrieve", tags=["Retriever"])

class RetrieveRequest(BaseModel):
    query: str
    context: ModelContext

@router.post("/")
async def retrieve(req: RetrieveRequest):
    vs = get_vectorstore(req.context.collection, req.context.namespace, req.context.embedding_model)
    results = vs.similarity_search_with_score(req.query, k=req.context.k)
    return [
        {"score": float(score), "metadata": doc.metadata, "content": doc.page_content}
        for doc, score in results
    ]
