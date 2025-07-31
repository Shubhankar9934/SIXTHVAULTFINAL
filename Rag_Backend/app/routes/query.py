from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
from app.services.rag import lightning_answer
from app.deps import get_current_user

router = APIRouter(tags=["query"])

class QueryIn(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None  # List of specific document IDs to search, None means all documents
    provider: Optional[str] = "ollama"  # AI provider selection
    model: Optional[str] = None  # Specific model selection

@router.post("/query")
async def query(
    body: QueryIn,
    hybrid: bool = Query(False),
    max_context: bool = Query(False, description="Use maximum context for very large documents"),
    user = Depends(get_current_user),
):
    # Use selected provider or default to Ollama
    selected_provider = body.provider or "ollama"
    
    # Use never-fail RAG with tenant-based access
    ans, src = await lightning_answer(
        user.tenant_id,  # Use tenant_id for tenant-based data sharing
        body.question, 
        hybrid, 
        selected_provider,  # Use user-selected provider
        max_context,
        document_ids=body.document_ids,
        selected_model=body.model  # Pass selected model
    )
    return {"answer": ans, "sources": src}
