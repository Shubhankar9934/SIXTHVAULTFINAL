# app/models_document_access.py
import uuid, datetime as dt
from typing import List, Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON

class DocumentAccess(SQLModel, table=True):
    """Document access control table for managing user permissions to documents"""
    __tablename__ = "document_access"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = Field(foreign_key="documents.id", index=True)
    user_id: str = Field(foreign_key="users.id", index=True)
    tenant_id: str = Field(foreign_key="tenants.id", index=True)  # Tenant isolation
    assigned_by: str = Field(foreign_key="users.id")  # Admin who assigned access
    
    # Permissions
    permissions: List[str] = Field(default_factory=lambda: ["read"], sa_column=Column(JSON))
    # Possible permissions: "read", "download", "search", "query"
    
    # Metadata
    assigned_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    expires_at: Optional[dt.datetime] = None  # Optional expiration
    is_active: bool = Field(default=True)
    
    # Audit trail
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    
    # Notes
    assignment_notes: Optional[str] = None

class UserDocumentGroup(SQLModel, table=True):
    """Groups of users for easier document assignment management"""
    __tablename__ = "user_document_groups"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str
    description: Optional[str] = None
    tenant_id: str = Field(foreign_key="tenants.id", index=True)
    created_by: str = Field(foreign_key="users.id")  # Admin who created the group
    
    # Group members
    user_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    
    # Metadata
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    is_active: bool = Field(default=True)

class DocumentAccessLog(SQLModel, table=True):
    """Audit log for document access events"""
    __tablename__ = "document_access_logs"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = Field(index=True)
    user_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    
    # Access details
    action: str  # "view", "download", "search", "query"
    access_granted: bool = Field(default=True)
    permission_used: str  # Which permission was used
    
    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    # Metadata
    timestamp: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    response_time_ms: Optional[int] = None
    
    # Denial reasons (if access_granted = False)
    denial_reason: Optional[str] = None
