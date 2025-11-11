# agents/supervisor.py
"""
Supervisor Agent for Hybrid RAG + SQL System
------------------------------------------------
Coordinates sub-agents:
- RAG Agent → document reasoning
- SQL Agent → query generation
- Supervisor → orchestrates both
"""

from langchain_core.tools import tool
from langchain.agents import create_agent
from helpers import get_vectorstore, load_llm
from context import ModelContext
import json, traceback

DEBUG = True
def log(msg): 
    if DEBUG: 
        print(f"[DEBUG] {msg}")


# -----------------------------------------------------------------------------
# 1. Base LLM for orchestration
# -----------------------------------------------------------------------------
orchestrator_llm = load_llm("qwen3:1.7b", temperature=0.2)


# -----------------------------------------------------------------------------
# 2. SQL Generation Utility
# -----------------------------------------------------------------------------
def run_sql_generation(question: str, dialect: str, llm_model: str) -> str:
    """Generate SQL using nearest schema chunks as context."""
    log(f"SQL Generation → dialect={dialect}, question={question}")

    vs = get_vectorstore("schemas", "global", "nomic-embed-text:latest")
    schema_docs = vs.similarity_search(question, k=5)
    schema_context = "\n\n".join(d.page_content for d in schema_docs)

    llm = load_llm(llm_model)
    prompt = f"""
You are an expert SQL assistant.
Use ONLY the following database schema context to generate the query.

Schema:
{schema_context}

Question:
{question}

Return only a valid {dialect} SQL statement. No explanations.
"""
    sql = llm.invoke(prompt)
    log(f"SQL Generated:\n{sql}")
    return str(sql)


# -----------------------------------------------------------------------------
# 3. Atomic Tools (RAG + SQL)
# -----------------------------------------------------------------------------
@tool
def rag_tool(query: str, context: dict | None = None) -> str:
    """Retrieve and answer questions from document embeddings."""
    try:
        ctx = ModelContext(**(context or {}))
        log(f"RAG Tool → query={query}, collection={ctx.collection}")

        vs = get_vectorstore(ctx.collection, ctx.namespace, ctx.embedding_model)
        docs = vs.similarity_search(query, k=ctx.k)
        context_text = "\n\n".join(d.page_content for d in docs[:3])

        llm = load_llm(ctx.llm_model, ctx.temperature)
        response = llm.invoke(f"Context:\n{context_text}\n\nUser: {query}")

        return f"{response}\n\nSources: {[d.metadata for d in docs]}"
    except Exception as e:
        log(traceback.format_exc())
        return f"[ERROR] RAG Tool failed: {e}"


@tool
def sql_tool(request_json: str) -> str:
    """Generate SQL query from natural language."""
    try:
        payload = json.loads(request_json)
    except json.JSONDecodeError:
        payload = {"question": request_json, "dialect": "sqlite"}

    sql = run_sql_generation(
        question=payload.get("question", ""),
        dialect=payload.get("dialect", "sqlite"),
        llm_model=payload.get("llm_model", "qwen3:1.7b")
    )
    result = {"sql": sql}
    log(f"sql_tool → returning: {result}")
    return json.dumps(result, ensure_ascii=False)


# -----------------------------------------------------------------------------
# 4. Sub-Agents (RAG + SQL Specialists)
# -----------------------------------------------------------------------------
rag_agent = create_agent(
    orchestrator_llm,
    tools=[rag_tool],
    system_prompt="You are the RAG Agent. Use rag_tool to answer document-based questions."
)

sql_agent = create_agent(
    orchestrator_llm,
    tools=[sql_tool],
    system_prompt="You are the SQL Agent. Use sql_tool to generate SQL from natural language."
)


# -----------------------------------------------------------------------------
# 5. Supervisor Agent (Orchestrator)
# -----------------------------------------------------------------------------
@tool
def run_rag_agent(request: str) -> str:
    log(f"run_rag_agent → {request}")
    result = rag_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


@tool
def run_sql_agent(request: str) -> str:
    log(f"run_sql_agent → {request}")
    result = sql_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


rag_sql_supervisor_agent = create_agent(
    orchestrator_llm,
    tools=[run_rag_agent, run_sql_agent],
    system_prompt=(
        "You are the Supervisor Agent for a hybrid RAG + SQL system.\n"
        "Use run_rag_agent for document or knowledge-base queries.\n"
        "Use run_sql_agent for analytical or structured-data questions.\n"
        "Return a clear, concise final answer."
    ),
)


def run_supervisor(query: str, ctx: ModelContext | None = None) -> str:
    """Main entrypoint for FastAPI."""
    ctx_json = (ctx or ModelContext()).model_dump_json()
    log(f"Supervisor → query='{query}' | context={ctx_json}")

    try:
        result = rag_sql_supervisor_agent.invoke({
            "messages": [
                {"role": "system", "content": f"ModelContext: {ctx_json}"},
                {"role": "user", "content": query},
            ]
        })
        return result["messages"][-1].text
    except Exception:
        log(traceback.format_exc())
        return "[ERROR] Supervisor failed"
