from fastapi import APIRouter
from pydantic import BaseModel
from helpers import load_llm

router = APIRouter(prefix="/sql", tags=["SQL"])

class SQLRequest(BaseModel):
    question: str
    dialect: str = "sqlite"
    llm_model: str = "qwen3:1.7b"

@router.post("/generate")
async def generate_sql(req: SQLRequest):
    llm = load_llm(req.llm_model)
    sql = llm.invoke(f"You are an expert SQL assistant. Generate a {req.dialect} query for: {req.question}")
    return {"query": sql}
