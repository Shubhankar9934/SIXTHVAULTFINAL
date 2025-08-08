# app/models.py
import uuid, datetime as dt
from typing import List, Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON   # ← import SQLAlchemy's JSON type
from app.database import User  # Import from database instead of auth.models

class Document(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="users.id")
    tenant_id: Optional[str] = Field(foreign_key="tenants.id")  # Add tenant isolation
    path: str
    filename: str  # Store original filename without UUID prefix

    # store the lists as a JSON column
    tags: List[str] | None = Field(default=None, sa_column=Column(JSON))
    demo_tags: List[str] | None = Field(default=None, sa_column=Column(JSON))

    summary: str | None = None
    insight: str | None = None
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

# AI Curation Models
class AICuration(SQLModel, table=True):
    __tablename__ = "ai_curations"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="users.id")
    tenant_id: Optional[str] = Field(foreign_key="tenants.id")  # Add tenant isolation
    title: str
    description: Optional[str] = None
    topic_keywords: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    document_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    confidence_score: float = Field(default=0.0)
    auto_generated: bool = Field(default=True)
    last_updated: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    status: str = Field(default="active")  # active, stale, archived, updating
    generation_method: str = Field(default="auto")  # auto, manual, incremental
    document_count: int = Field(default=0)
    relevance_threshold: float = Field(default=0.5)
    content: Optional[str] = None  # Store the generated curation content
    content_generated: bool = Field(default=False)  # Track if content has been generated
    provider_used: Optional[str] = None  # Track which AI provider was used
    model_used: Optional[str] = None  # Track which model was used

class CurationSettings(SQLModel, table=True):
    __tablename__ = "curation_settings"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="users.id", unique=True)
    auto_refresh: bool = Field(default=True)
    on_add: str = Field(default="incremental")  # full, incremental, manual
    on_delete: str = Field(default="auto_clean")  # auto_clean, keep_stale, prompt
    change_threshold: int = Field(default=15)  # % docs changed before auto-update
    max_curations: int = Field(default=8)
    min_documents_per_curation: int = Field(default=2)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class DocumentCurationMapping(SQLModel, table=True):
    __tablename__ = "document_curation_mapping"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str
    curation_id: str = Field(foreign_key="ai_curations.id")
    owner_id: str = Field(foreign_key="users.id")
    relevance_score: float = Field(default=0.0)
    keywords_matched: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class CurationGenerationHistory(SQLModel, table=True):
    __tablename__ = "curation_generation_history"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="users.id")
    generation_type: str  # full, incremental, manual, cleanup
    trigger_event: Optional[str] = None  # document_added, document_deleted, manual_refresh
    documents_processed: int = Field(default=0)
    curations_created: int = Field(default=0)
    curations_updated: int = Field(default=0)
    curations_deleted: int = Field(default=0)
    processing_time_ms: int = Field(default=0)
    success: bool = Field(default=True)
    error_message: Optional[str] = None
    meta_data: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

# AI Summary Models
class AISummary(SQLModel, table=True):
    __tablename__ = "ai_summaries"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="users.id")
    tenant_id: Optional[str] = Field(foreign_key="tenants.id")  # Add tenant isolation
    title: str
    description: Optional[str] = None
    summary_type: str = Field(default="auto")  # auto, individual, combined, custom
    focus_keywords: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    document_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    confidence_score: float = Field(default=0.0)
    auto_generated: bool = Field(default=True)
    last_updated: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    status: str = Field(default="active")  # active, stale, archived, updating
    generation_method: str = Field(default="auto")  # auto, manual, incremental
    document_count: int = Field(default=0)
    content_length: int = Field(default=0)
    summary_content: Optional[str] = None  # Store the actual summary content

class SummarySettings(SQLModel, table=True):
    __tablename__ = "summary_settings"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="users.id", unique=True)
    auto_refresh: bool = Field(default=True)
    on_add: str = Field(default="incremental")  # full, incremental, manual
    on_delete: str = Field(default="auto_clean")  # auto_clean, keep_stale, prompt
    change_threshold: int = Field(default=15)  # % docs changed before auto-update
    max_summaries: int = Field(default=8)
    min_documents_per_summary: int = Field(default=1)
    default_summary_type: str = Field(default="auto")  # auto, individual, combined
    include_individual: bool = Field(default=True)
    include_combined: bool = Field(default=True)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class DocumentSummaryMapping(SQLModel, table=True):
    __tablename__ = "document_summary_mapping"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str
    summary_id: str = Field(foreign_key="ai_summaries.id")
    owner_id: str = Field(foreign_key="users.id")
    relevance_score: float = Field(default=0.0)
    contribution_weight: float = Field(default=1.0)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class SummaryGenerationHistory(SQLModel, table=True):
    __tablename__ = "summary_generation_history"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="users.id")
    generation_type: str  # full, incremental, manual, cleanup
    trigger_event: Optional[str] = None  # document_added, document_deleted, manual_refresh
    documents_processed: int = Field(default=0)
    summaries_created: int = Field(default=0)
    summaries_updated: int = Field(default=0)
    summaries_deleted: int = Field(default=0)
    processing_time_ms: int = Field(default=0)
    success: bool = Field(default=True)
    error_message: Optional[str] = None
    meta_data: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

# Chat History Models - NEW
class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="users.id")
    tenant_id: Optional[str] = Field(foreign_key="tenants.id")  # Add tenant isolation
    title: str  # Auto-generated from first message or user-defined
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    message_count: int = Field(default=0)
    is_archived: bool = Field(default=False)
    is_pinned: bool = Field(default=False)
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    
    # Conversation metadata
    total_tokens_used: int = Field(default=0)
    primary_provider: Optional[str] = None  # Most used provider in this conversation
    primary_model: Optional[str] = None  # Most used model in this conversation
    document_ids_referenced: List[str] = Field(default_factory=list, sa_column=Column(JSON))

class Message(SQLModel, table=True):
    __tablename__ = "messages"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    conversation_id: str = Field(foreign_key="conversations.id")
    role: str  # "user", "assistant", "curation", "summary"
    content: str
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    
    # RAG-specific fields
    provider_used: Optional[str] = None
    model_used: Optional[str] = None
    document_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    sources: List[dict] = Field(default_factory=list, sa_column=Column(JSON))
    context_mode: Optional[str] = None  # "hybrid", "pure_rag"
    
    # Message metadata
    message_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    tokens_used: Optional[int] = None
    response_time_ms: Optional[int] = None
    
    # For curations and summaries
    curation_id: Optional[str] = None  # Link to AICuration if this is a curation message
    summary_id: Optional[str] = None   # Link to AISummary if this is a summary message
    
    # Message status
    is_edited: bool = Field(default=False)
    edit_history: List[dict] = Field(default_factory=list, sa_column=Column(JSON))

class ConversationSettings(SQLModel, table=True):
    __tablename__ = "conversation_settings"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="users.id", unique=True)
    
    # Auto-save settings
    auto_save_conversations: bool = Field(default=True)
    auto_generate_titles: bool = Field(default=True)
    max_conversations: int = Field(default=100)
    auto_archive_after_days: int = Field(default=30)
    
    # Default conversation behavior
    default_provider: str = Field(default="gemini")
    default_model: str = Field(default="gemini-1.5-flash")
    save_curation_conversations: bool = Field(default=True)
    save_summary_conversations: bool = Field(default=True)
    
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class ProcessingDocument(SQLModel, table=True):
    """Database-backed processing tracker for user isolation and persistence"""
    __tablename__ = "processing_documents"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    doc_id: str = Field(unique=True, index=True)  # The processing document ID
    filename: str
    batch_id: str = Field(index=True)
    owner_id: str = Field(foreign_key="users.id", index=True)
    
    # Processing state
    status: str = Field(default="queued")  # queued, uploading, processing, completed, cancelled, error
    progress: int = Field(default=0)
    stage: str = Field(default="queued")
    error_message: Optional[str] = None
    
    # File information
    file_path: Optional[str] = None  # Local temp file path
    s3_key: Optional[str] = None     # S3 object key
    s3_bucket: Optional[str] = None  # S3 bucket name
    file_size: Optional[int] = None
    content_type: Optional[str] = None
    
    # Timestamps
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    started_at: Optional[dt.datetime] = None
    completed_at: Optional[dt.datetime] = None
    
    # Control flags
    cancellation_requested: bool = Field(default=False)
    cleanup_completed: bool = Field(default=False)
