# app/routes/conversations.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.deps import get_current_user, get_current_user_optional
from app.database import User
from app.services.conversation_service import conversation_service
from app.models import Conversation, Message, ConversationSettings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])

# Simple health check route for debugging
@router.get("/health")
async def conversations_health():
    """Health check for conversations router"""
    return {"status": "healthy", "message": "Conversations router is working"}

# Debug route to test basic functionality
@router.get("/debug")
async def conversations_debug():
    """Debug endpoint to test basic router functionality"""
    return {"status": "debug", "message": "Debug endpoint working", "endpoint": "/conversations/debug"}

# Test route without dependencies
@router.get("/test-no-deps")
async def test_no_dependencies():
    """Test endpoint without any dependencies"""
    return {"status": "success", "message": "No dependencies endpoint working", "data": []}


# Request/Response Models
class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    auto_generate_title: bool = True

class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None
    is_archived: Optional[bool] = None
    is_pinned: Optional[bool] = None
    tags: Optional[List[str]] = None

class AddMessageRequest(BaseModel):
    role: str  # "user", "assistant", "curation", "summary"
    content: str
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    document_ids: Optional[List[str]] = None
    sources: Optional[List[Dict]] = None
    context_mode: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None
    response_time_ms: Optional[int] = None
    curation_id: Optional[str] = None
    summary_id: Optional[str] = None

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    is_archived: bool
    is_pinned: bool
    tags: List[str]
    total_tokens_used: int
    primary_provider: Optional[str]
    primary_model: Optional[str]
    document_ids_referenced: List[str]

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    provider_used: Optional[str]
    model_used: Optional[str]
    document_ids: List[str]
    sources: List[Dict]
    context_mode: Optional[str]
    metadata: Dict[str, Any]
    tokens_used: Optional[int]
    response_time_ms: Optional[int]
    curation_id: Optional[str]
    summary_id: Optional[str]
    is_edited: bool

class ConversationWithMessagesResponse(BaseModel):
    conversation: ConversationResponse
    messages: List[MessageResponse]

class SearchResult(BaseModel):
    conversation: ConversationResponse
    matching_messages: List[MessageResponse]

class ConversationStatsResponse(BaseModel):
    total_conversations: int
    active_conversations: int
    archived_conversations: int
    total_messages: int
    total_tokens_used: int

# Helper functions
def conversation_to_response(conv: Conversation) -> ConversationResponse:
    """Convert Conversation model to response"""
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at.isoformat(),
        updated_at=conv.updated_at.isoformat(),
        message_count=conv.message_count,
        is_archived=conv.is_archived,
        is_pinned=conv.is_pinned,
        tags=conv.tags,
        total_tokens_used=conv.total_tokens_used,
        primary_provider=conv.primary_provider,
        primary_model=conv.primary_model,
        document_ids_referenced=conv.document_ids_referenced
    )

def message_to_response(msg: Message) -> MessageResponse:
    """Convert Message model to response"""
    return MessageResponse(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at.isoformat(),
        provider_used=msg.provider_used,
        model_used=msg.model_used,
        document_ids=msg.document_ids,
        sources=msg.sources,
        context_mode=msg.context_mode,
        metadata=msg.message_metadata,
        tokens_used=msg.tokens_used,
        response_time_ms=msg.response_time_ms,
        curation_id=msg.curation_id,
        summary_id=msg.summary_id,
        is_edited=msg.is_edited
    )

# Routes
@router.post("/", response_model=ConversationResponse)
@router.post("", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new conversation"""
    try:
        conversation = conversation_service.create_conversation(
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            title=request.title,
            auto_generate_title=request.auto_generate_title
        )
        
        logger.info(f"Created conversation {conversation.id} for user {current_user.id}")
        return conversation_to_response(conversation)
        
    except Exception as e:
        logger.error(f"Error creating conversation for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")

@router.get("/", response_model=List[ConversationResponse])
@router.get("", response_model=List[ConversationResponse])
async def get_conversations(
    include_archived: bool = Query(False, description="Include archived conversations"),
    limit: int = Query(50, ge=1, le=100, description="Number of conversations to return"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip"),
    search: Optional[str] = Query(None, description="Search query for conversation titles"),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get user's conversations"""
    try:
        # If user is not authenticated, return empty list
        if not current_user:
            logger.info("Unauthenticated request to conversations endpoint, returning empty list")
            return []
        
        logger.info(f"Fetching conversations for user {current_user.id}")
        conversations = conversation_service.get_user_conversations(
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            include_archived=include_archived,
            limit=limit,
            offset=offset,
            search_query=search
        )
        
        logger.info(f"Found {len(conversations)} conversations for user {current_user.id}")
        return [conversation_to_response(conv) for conv in conversations]
        
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        # Return empty list instead of raising exception for better UX
        return []

@router.get("/{conversation_id}", response_model=ConversationWithMessagesResponse)
async def get_conversation(
    conversation_id: str,
    message_limit: int = Query(100, ge=1, le=500, description="Number of messages to return"),
    message_offset: int = Query(0, ge=0, description="Number of messages to skip"),
    current_user: User = Depends(get_current_user)
):
    """Get a specific conversation with its messages"""
    try:
        # Get conversation
        conversation = conversation_service.get_conversation_by_id(
            conversation_id=conversation_id,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages
        messages = conversation_service.get_conversation_messages(
            conversation_id=conversation_id,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            limit=message_limit,
            offset=message_offset
        )
        
        return ConversationWithMessagesResponse(
            conversation=conversation_to_response(conversation),
            messages=[message_to_response(msg) for msg in messages]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation {conversation_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation")

@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    current_user: User = Depends(get_current_user)
):
    """Update conversation properties"""
    try:
        conversation = conversation_service.update_conversation(
            conversation_id=conversation_id,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            title=request.title,
            is_archived=request.is_archived,
            is_pinned=request.is_pinned,
            tags=request.tags
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Updated conversation {conversation_id} for user {current_user.id}")
        return conversation_to_response(conversation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation {conversation_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update conversation")

@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a conversation and all its messages"""
    try:
        success = conversation_service.delete_conversation(
            conversation_id=conversation_id,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Deleted conversation {conversation_id} for user {current_user.id}")
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(
    conversation_id: str,
    request: AddMessageRequest,
    current_user: User = Depends(get_current_user)
):
    """Add a message to a conversation"""
    try:
        message = conversation_service.add_message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            provider_used=request.provider_used,
            model_used=request.model_used,
            document_ids=request.document_ids,
            sources=request.sources,
            context_mode=request.context_mode,
            metadata=request.metadata,
            tokens_used=request.tokens_used,
            response_time_ms=request.response_time_ms,
            curation_id=request.curation_id,
            summary_id=request.summary_id
        )
        
        if not message:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Added message to conversation {conversation_id} for user {current_user.id}")
        return message_to_response(message)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message to conversation {conversation_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add message")

@router.get("/search/content", response_model=List[SearchResult])
async def search_conversations(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=50, description="Number of results to return"),
    current_user: User = Depends(get_current_user)
):
    """Search conversations by content"""
    try:
        results = conversation_service.search_conversations(
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            query=q,
            limit=limit
        )
        
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                conversation=conversation_to_response(result["conversation"]),
                matching_messages=[message_to_response(msg) for msg in result["matching_messages"]]
            ))
        
        return search_results
        
    except Exception as e:
        logger.error(f"Error searching conversations for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to search conversations")

@router.get("/stats/overview", response_model=ConversationStatsResponse)
async def get_conversation_stats(
    current_user: User = Depends(get_current_user)
):
    """Get conversation statistics for the user"""
    try:
        stats = conversation_service.get_conversation_stats(
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        return ConversationStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error fetching conversation stats for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation statistics")

@router.get("/settings/preferences", response_model=ConversationSettings)
async def get_conversation_settings(
    current_user: User = Depends(get_current_user)
):
    """Get user's conversation settings"""
    try:
        settings = conversation_service.get_conversation_settings(
            user_id=current_user.id
        )
        
        return settings
        
    except Exception as e:
        logger.error(f"Error fetching conversation settings for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation settings")

@router.put("/settings/preferences", response_model=ConversationSettings)
async def update_conversation_settings(
    settings_update: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Update user's conversation settings"""
    try:
        settings = conversation_service.update_conversation_settings(
            user_id=current_user.id,
            **settings_update
        )
        
        logger.info(f"Updated conversation settings for user {current_user.id}")
        return settings
        
    except Exception as e:
        logger.error(f"Error updating conversation settings for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update conversation settings")

@router.post("/cleanup/archive")
async def cleanup_old_conversations(
    days_old: int = Query(30, ge=1, le=365, description="Archive conversations older than this many days"),
    current_user: User = Depends(get_current_user)
):
    """Archive old conversations based on age"""
    try:
        count = conversation_service.cleanup_old_conversations(
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            days_old=days_old
        )
        
        logger.info(f"Archived {count} old conversations for user {current_user.id}")
        return {"message": f"Archived {count} conversations", "count": count}
        
    except Exception as e:
        logger.error(f"Error cleaning up conversations for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup conversations")
