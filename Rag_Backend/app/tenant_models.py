# app/tenant_models.py
import uuid
import datetime as dt
from typing import List, Optional, Dict, Any
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON

class Tenant(SQLModel, table=True):
    """Enhanced tenant model for multi-tenant architecture"""
    __tablename__ = "tenants"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    slug: Optional[str] = Field(default=None, unique=True, index=True)  # 'pepsi', 'kfc', 'airtel', 'sapienplus', 'public'
    name: str  # Display name
    tenant_type: str = Field(default="enterprise")  # 'enterprise', 'partner', 'public'
    is_active: bool = Field(default=True)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    
    # Tenant-specific configurations
    max_users: Optional[int] = None
    max_storage_gb: Optional[int] = None
    max_documents: Optional[int] = None
    allowed_file_types: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    
    # Branding
    logo_url: Optional[str] = None
    primary_color: Optional[str] = None  # Hex color
    secondary_color: Optional[str] = None  # Hex color
    custom_domain: Optional[str] = None
    
    # Features enabled for this tenant
    features: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Tenant settings
    settings: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Owner information
    owner_id: Optional[str] = Field(foreign_key="users.id")
    
    # Usage tracking
    current_users: int = Field(default=0)
    current_storage_mb: int = Field(default=0)
    current_documents: int = Field(default=0)
    
    # Billing information (for future use)
    subscription_tier: Optional[str] = None
    billing_email: Optional[str] = None
    
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class TenantUser(SQLModel, table=True):
    """Junction table for tenant-user relationships with roles"""
    __tablename__ = "tenant_users"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id")
    user_id: str = Field(foreign_key="users.id")
    
    # Tenant-specific role
    tenant_role: str = Field(default="user")  # 'admin', 'manager', 'user', 'viewer'
    
    # User status within tenant
    status: str = Field(default="active")  # 'active', 'inactive', 'suspended', 'pending'
    
    # Permissions within tenant
    permissions: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    
    # User limits within tenant
    storage_limit_mb: int = Field(default=100)
    storage_used_mb: int = Field(default=0)
    document_limit: int = Field(default=10)
    document_count: int = Field(default=0)
    
    # Timestamps
    joined_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    last_active: Optional[dt.datetime] = None
    
    # Invitation tracking
    invited_by: Optional[str] = Field(foreign_key="users.id")
    invitation_accepted_at: Optional[dt.datetime] = None

class TenantInvitation(SQLModel, table=True):
    """Tenant invitation system"""
    __tablename__ = "tenant_invitations"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id")
    email: str
    tenant_role: str = Field(default="user")
    
    # Invitation details
    invited_by: str = Field(foreign_key="users.id")
    invitation_token: str = Field(unique=True)
    expires_at: dt.datetime
    
    # Status tracking
    status: str = Field(default="pending")  # 'pending', 'accepted', 'expired', 'cancelled'
    accepted_by: Optional[str] = Field(foreign_key="users.id")
    accepted_at: Optional[dt.datetime] = None
    
    # Metadata
    invitation_message: Optional[str] = None
    permissions: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class TenantSettings(SQLModel, table=True):
    """Tenant-specific settings and configurations"""
    __tablename__ = "tenant_settings"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id", unique=True)
    
    # Document management settings
    allow_user_uploads: bool = Field(default=True)
    require_admin_approval: bool = Field(default=False)
    auto_process_documents: bool = Field(default=True)
    
    # AI settings
    ai_provider: str = Field(default="groq")
    ai_model: str = Field(default="llama-3.1-8b-instant")
    ai_temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    
    # Feature flags
    enable_advanced_analytics: bool = Field(default=False)
    enable_custom_prompts: bool = Field(default=False)
    enable_api_access: bool = Field(default=False)
    enable_white_labeling: bool = Field(default=False)
    enable_bulk_operations: bool = Field(default=False)
    
    # Security settings
    enforce_2fa: bool = Field(default=False)
    session_timeout_minutes: int = Field(default=480)  # 8 hours
    password_policy: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Notification settings
    email_notifications: bool = Field(default=True)
    slack_webhook: Optional[str] = None
    notification_preferences: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Custom configurations
    custom_fields: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class TenantAnalytics(SQLModel, table=True):
    """Tenant usage analytics and metrics"""
    __tablename__ = "tenant_analytics"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id")
    
    # Date for the analytics record
    date: dt.date = Field(index=True)
    
    # User metrics
    active_users: int = Field(default=0)
    new_users: int = Field(default=0)
    total_users: int = Field(default=0)
    
    # Document metrics
    documents_uploaded: int = Field(default=0)
    documents_processed: int = Field(default=0)
    total_documents: int = Field(default=0)
    storage_used_mb: int = Field(default=0)
    
    # AI usage metrics
    ai_queries: int = Field(default=0)
    ai_tokens_used: int = Field(default=0)
    curations_generated: int = Field(default=0)
    summaries_generated: int = Field(default=0)
    
    # Performance metrics
    avg_response_time_ms: Optional[float] = None
    error_rate: float = Field(default=0.0)
    uptime_percentage: float = Field(default=100.0)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

class TenantAuditLog(SQLModel, table=True):
    """Audit log for tenant activities"""
    __tablename__ = "tenant_audit_logs"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.id")
    user_id: Optional[str] = Field(foreign_key="users.id")
    
    # Action details
    action: str  # 'user_added', 'document_uploaded', 'settings_changed', etc.
    resource_type: str  # 'user', 'document', 'settings', 'tenant'
    resource_id: Optional[str] = None
    
    # Action metadata
    details: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Result
    success: bool = Field(default=True)
    error_message: Optional[str] = None
    
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)

# Default tenant configurations
DEFAULT_TENANT_CONFIGS = {
    "enterprise": {
        "features": {
            "advanced_analytics": True,
            "custom_prompts": True,
            "api_access": True,
            "white_labeling": True,
            "bulk_operations": True
        },
        "max_users": 1000,
        "max_storage_gb": 2000,
        "max_documents": 10000,
        "allowed_file_types": ["pdf", "docx", "txt", "md", "csv", "xlsx", "pptx"]
    },
    "partner": {
        "features": {
            "advanced_analytics": True,
            "custom_prompts": True,
            "api_access": True,
            "white_labeling": False,
            "bulk_operations": True
        },
        "max_users": 200,
        "max_storage_gb": 500,
        "max_documents": 2000,
        "allowed_file_types": ["pdf", "docx", "txt", "md", "csv"]
    },
    "public": {
        "features": {
            "advanced_analytics": False,
            "custom_prompts": False,
            "api_access": False,
            "white_labeling": False,
            "bulk_operations": False
        },
        "max_users": 10000,
        "max_storage_gb": 100,  # Total for all users
        "max_documents": 1000,  # Total for all users
        "allowed_file_types": ["pdf", "docx", "txt"]
    }
}

# Initial tenant data
INITIAL_TENANTS = [
    {
        "slug": "pepsi",
        "name": "PepsiCo",
        "tenant_type": "enterprise",
        "primary_color": "#004B93",
        "secondary_color": "#E32934"
    },
    {
        "slug": "kfc",
        "name": "KFC Corporation", 
        "tenant_type": "enterprise",
        "primary_color": "#E4002B",
        "secondary_color": "#FFC72C"
    },
    {
        "slug": "airtel",
        "name": "Bharti Airtel",
        "tenant_type": "enterprise", 
        "primary_color": "#E60012",
        "secondary_color": "#000000"
    },
    {
        "slug": "sapienplus",
        "name": "SapienPlus.ai",
        "tenant_type": "partner",
        "primary_color": "#6366F1",
        "secondary_color": "#8B5CF6"
    },
    {
        "slug": "public",
        "name": "Public Access",
        "tenant_type": "public",
        "primary_color": "#3B82F6",
        "secondary_color": "#6366F1"
    }
]
