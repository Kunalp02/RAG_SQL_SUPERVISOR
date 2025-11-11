# main.py
"""
Main FastAPI Entrypoint
------------------------
Integrates RAG ingestion, SQL generation, retrieval, and chat (multi-agent) endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import rag_routes, sql_routes, retrieve_routes, chat_routes

app = FastAPI(
    title="🧠 RAG + SQL + Chat API",
    version="1.0.0",
    description="Unified backend for document retrieval (RAG), SQL generation, and chat-based orchestration."
)

# -----------------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------
app.include_router(rag_routes.router)
app.include_router(sql_routes.router)
app.include_router(retrieve_routes.router)
app.include_router(chat_routes.router)

# -----------------------------------------------------------------------------
# Root Health Endpoint
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "🧠 RAG + SQL + Chat API is running successfully!"}

# -----------------------------------------------------------------------------
# Local Dev Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
