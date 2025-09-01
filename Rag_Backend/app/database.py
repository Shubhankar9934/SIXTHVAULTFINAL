from sqlmodel import create_engine, Session, SQLModel, Field
from sqlalchemy import text, Column, JSON
from typing import Generator, Optional, List
from datetime import datetime
import uuid
from app.config import settings

def get_database_url():
    """Get PostgreSQL database URL"""
    return f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

DATABASE_URL = get_database_url()

# Create PostgreSQL engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=10,
    max_overflow=20
)

class TempUser(SQLModel, table=True):
    __tablename__ = "temp_users"
    
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    email: str
    password_hash: str
    first_name: str
    last_name: str
    company: Optional[str] = None
    verification_code: str
    verification_code_expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class UserBase(SQLModel):
    email: str
    username: Optional[str] = None
    first_name: str
    last_name: str
    company: Optional[str] = None
    company_id: Optional[str] = None
    verified: bool = False
    role: str = "user"
    is_admin: bool = False
    is_super_admin: bool = False  # New: Super admin for cross-tenant management
    is_active: bool = True
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    # Multi-tenant fields
    tenant_id: Optional[str] = Field(foreign_key="tenants.id")  # User's current tenant context
    primary_tenant_id: Optional[str] = Field(foreign_key="tenants.id")  # User's primary tenant
    subscription_tier: str = Field(default="free")  # For public tenant users
    storage_used_mb: int = Field(default=0)
    storage_limit_mb: int = Field(default=100)

class User(UserBase, table=True):
    __tablename__ = "users"
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    password_hash: str
    reset_token: Optional[str] = None
    reset_token_expires_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None  # Track who created this user (for admin-created users)

class UserToken(SQLModel, table=True):
    __tablename__ = "user_tokens"
    
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(foreign_key="users.id")
    token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    device_info: Optional[str] = None
    last_used: Optional[datetime] = None


def get_session() -> Generator[Session, None, None]:
    """Get a new database session"""
    with Session(engine) as session:
        yield session

def init_db():
    """Initialize database by creating tables if they don't exist"""
    try:
        # Import all models to ensure they're included in metadata
        from app.models import (
            Document, ProcessingDocument, AICuration, CurationSettings,
            DocumentCurationMapping, CurationGenerationHistory, AISummary,
            SummarySettings, DocumentSummaryMapping, SummaryGenerationHistory,
            Conversation, Message, ConversationSettings
        )
        from app.models_document_access import (
            DocumentAccess, UserDocumentGroup, DocumentAccessLog
        )
        from app.tenant_models import (
            Tenant, TenantUser, TenantInvitation, TenantSettings,
            TenantAnalytics, TenantAuditLog
        )
        
        # Create tables only if they don't exist (PostgreSQL handles IF NOT EXISTS automatically)
        SQLModel.metadata.create_all(engine)
        print("Database tables initialized successfully")
    except Exception as e:
        print(f"Error during database initialization: {e}")
        raise

# Import Tenant model for use in other modules
from app.tenant_models import Tenant

def reset_database():
    """Reset database by dropping and recreating all tables - USE WITH CAUTION"""
    try:
        # Import all models to ensure they're included in metadata
        from app.models import (
            Document, ProcessingDocument, AICuration, CurationSettings,
            DocumentCurationMapping, CurationGenerationHistory, AISummary,
            SummarySettings, DocumentSummaryMapping, SummaryGenerationHistory,
            Conversation, Message, ConversationSettings
        )
        from app.models_document_access import (
            DocumentAccess, UserDocumentGroup, DocumentAccessLog
        )
        from app.tenant_models import (
            Tenant, TenantUser, TenantInvitation, TenantSettings,
            TenantAnalytics, TenantAuditLog
        )
        
        # Drop all tables with CASCADE to handle foreign key constraints
        with engine.begin() as conn:
            # Drop document access tables first
            conn.execute(text("DROP TABLE IF EXISTS document_access_logs CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS user_document_groups CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS document_access CASCADE"))
            
            # Drop tenant-related tables
            conn.execute(text("DROP TABLE IF EXISTS tenant_audit_logs CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS tenant_analytics CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS tenant_settings CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS tenant_invitations CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS tenant_users CASCADE"))
            
            # Drop existing tables in reverse dependency order
            conn.execute(text("DROP TABLE IF EXISTS conversation_settings CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS messages CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS conversations CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS summary_generation_history CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS document_summary_mapping CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS summary_settings CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS ai_summaries CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS curation_generation_history CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS document_curation_mapping CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS curation_settings CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS ai_curations CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS processing_documents CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS user_tokens CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS documents CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS temp_users CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS tenants CASCADE"))
        print("Dropped existing tables")
        
        # Create new tables
        SQLModel.metadata.create_all(engine)
        print("Created new database tables")
    except Exception as e:
        print(f"Error during database reset: {e}")
        raise
