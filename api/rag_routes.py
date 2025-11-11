from fastapi import APIRouter, UploadFile, File, Form
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
)
from langchain_core.documents import Document
from context import ModelContext
from helpers import get_vectorstore
import json, os

router = APIRouter(prefix="/rag", tags=["RAG"])

# ------------------------------------------------------------
# Document Ingestion
# ------------------------------------------------------------
@router.post("/ingest-document")
async def ingest_document(file: UploadFile = File(...), context: str = Form(...)):
    ctx = ModelContext(**json.loads(context))
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
    }
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in loaders:
        return {"error": f"Unsupported file type: {ext}"}

    docs = loaders[ext](temp_path).load()
    for d in docs: d.metadata.update(ctx.metadata)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=ctx.chunk_size, chunk_overlap=ctx.chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    get_vectorstore(ctx.collection, ctx.namespace, ctx.embedding_model).add_documents(chunks)

    os.remove(temp_path)
    return {"status": "success", "chunks": len(chunks), "collection": ctx.collection}


# ------------------------------------------------------------
# Schema Ingestion
# ------------------------------------------------------------
def resolve_db_uri(db_name: str) -> str:
    """Map a logical DB name to a real connection URI."""
    registry_path = "db_registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
            if db_name in registry:
                return registry[db_name]
    os.makedirs("data", exist_ok=True)
    return f"sqlite:///{os.path.abspath(os.path.join('data', f'{db_name}.db'))}"


@router.post("/ingest-schema")
async def ingest_schema(file: UploadFile = File(...), context: str = Form(...)):
    ctx = ModelContext(**json.loads(context))
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    text = open(temp_path, "r", encoding="utf-8").read()
    os.remove(temp_path)

    db_name = next(
        (line.split(":", 1)[1].strip() for line in text.splitlines() if line.lower().startswith("database:")),
        "unknown_db"
    )
    db_uri = resolve_db_uri(db_name)

    sections = text.split("Table: ")[1:]
    chunks = [
        Document(
            page_content="Table: " + s,
            metadata={"db": db_name, "table": s.split("\n", 1)[0].strip(), "uri": db_uri, "type": "schema"},
        )
        for s in sections
    ]

    vs = get_vectorstore("schemas", ctx.namespace, ctx.embedding_model)
    vs.add_documents(chunks)
    return {"status": "success", "db": db_name, "tables": len(chunks), "uri": db_uri, "collection": "schemas"}
