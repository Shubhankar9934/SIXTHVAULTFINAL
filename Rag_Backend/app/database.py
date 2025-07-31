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

class Tenant(SQLModel, table=True):
    __tablename__ = "tenants"
    
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str  # Company/Organization name
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    owner_id: Optional[str] = None  # The user who owns this tenant

class UserBase(SQLModel):
    email: str
    username: Optional[str] = None
    first_name: str
    last_name: str
    company: Optional[str] = None
    company_id: Optional[str] = None
    tenant_id: Optional[str] = None  # Add tenant isolation
    verified: bool = False
    role: str = "user"
    is_admin: bool = False
    is_active: bool = True
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

class User(UserBase, table=True):
    __tablename__ = "users"
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    password_hash: str
    reset_token: Optional[str] = None
    reset_token_expires_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None  # Track who created this user (for admin-created users)
    tenant_id: Optional[str] = Field(foreign_key="tenants.id")  # Link to tenant

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
            SummarySettings, DocumentSummaryMapping, SummaryGenerationHistory
        )
        
        # Create tables only if they don't exist (PostgreSQL handles IF NOT EXISTS automatically)
        SQLModel.metadata.create_all(engine)
        print("Database tables initialized successfully")
    except Exception as e:
        print(f"Error during database initialization: {e}")
        raise

def reset_database():
    """Reset database by dropping and recreating all tables - USE WITH CAUTION"""
    try:
        # Import all models to ensure they're included in metadata
        from app.models import (
            Document, ProcessingDocument, AICuration, CurationSettings,
            DocumentCurationMapping, CurationGenerationHistory, AISummary,
            SummarySettings, DocumentSummaryMapping, SummaryGenerationHistory
        )
        
        # Drop all tables with CASCADE to handle foreign key constraints
        with engine.begin() as conn:
            # Drop all tables in reverse dependency order
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
            conn.execute(text("DROP TABLE IF EXISTS document CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS temp_users CASCADE"))
        print("Dropped existing tables")
        
        # Create new tables
        SQLModel.metadata.create_all(engine)
        print("Created new database tables")
    except Exception as e:
        print(f"Error during database reset: {e}")
        raise
