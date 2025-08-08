from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
import time
from app.services.rag import lightning_answer
from app.services.conversation_service import conversation_service
from app.deps import get_current_user

router = APIRouter(tags=["query"])

class QueryIn(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None  # List of specific document IDs to search, None means all documents
    provider: Optional[str] = "ollama"  # AI provider selection
    model: Optional[str] = None  # Specific model selection
    conversation_id: Optional[str] = None  # Optional conversation ID to continue existing conversation
    save_conversation: bool = True  # Whether to save this query as a conversation

@router.post("/query")
async def query(
    body: QueryIn,
    hybrid: bool = Query(False),
    max_context: bool = Query(False, description="Use maximum context for very large documents"),
    user = Depends(get_current_user),
):
    start_time = time.time()
    
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
    
    response_time_ms = int((time.time() - start_time) * 1000)
    
    # Auto-save conversation if enabled
    conversation_id = None
    if body.save_conversation:
        try:
            # Get user's conversation settings
            settings = conversation_service.get_conversation_settings(user.id)
            
            if settings.auto_save_conversations:
                # Create or get existing conversation
                if body.conversation_id:
                    # Continue existing conversation
                    conversation = conversation_service.get_conversation_by_id(
                        body.conversation_id, user.id, user.tenant_id
                    )
                    if conversation:
                        conversation_id = conversation.id
                
                if not conversation_id:
                    # Create new conversation
                    conversation = conversation_service.create_conversation(
                        user_id=user.id,
                        tenant_id=user.tenant_id,
                        title=None,  # Will be auto-generated from first message
                        auto_generate_title=settings.auto_generate_titles
                    )
                    conversation_id = conversation.id
                
                # Add user message
                conversation_service.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=body.question,
                    user_id=user.id,
                    tenant_id=user.tenant_id,
                    provider_used=selected_provider,
                    model_used=body.model,
                    document_ids=body.document_ids,
                    context_mode="hybrid" if hybrid else "pure_rag",
                    metadata={
                        "max_context": max_context,
                        "query_type": "rag_search"
                    }
                )
                
                # Add assistant response
                conversation_service.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=ans,
                    user_id=user.id,
                    tenant_id=user.tenant_id,
                    provider_used=selected_provider,
                    model_used=body.model,
                    document_ids=body.document_ids,
                    sources=src,
                    context_mode="hybrid" if hybrid else "pure_rag",
                    metadata={
                        "max_context": max_context,
                        "query_type": "rag_response",
                        "source_count": len(src) if src else 0
                    },
                    response_time_ms=response_time_ms
                )
                
        except Exception as e:
            # Log error but don't fail the query
            print(f"Failed to save conversation for user {user.id}: {e}")
    
    response = {"answer": ans, "sources": src}
    
    # Include conversation_id in response if conversation was saved
    if conversation_id:
        response["conversation_id"] = conversation_id
    
    return response
