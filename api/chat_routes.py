from fastapi import APIRouter
from pydantic import BaseModel
from context import ModelContext
from agents.supervisor import run_supervisor

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    query: str
    context: ModelContext | None = None

@router.post("/")
async def chat(req: ChatRequest):
    """Main conversational endpoint for the Supervisor Agent."""
    response = run_supervisor(req.query, req.context)
    return {"response": response}
