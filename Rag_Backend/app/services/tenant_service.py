# app/services/tenant_service.py
from typing import List, Optional, Dict, Any
from sqlmodel import Session, select
from datetime import datetime, timedelta
import uuid

from app.database import get_session
from app.tenant_models import (
    Tenant, TenantUser, TenantInvitation, TenantSettings,
    TenantAnalytics, TenantAuditLog, DEFAULT_TENANT_CONFIGS, INITIAL_TENANTS
)
from app.database import User


class TenantService:
    """Service for managing multi-tenant operations"""
    
    def __init__(self):
        pass
    
    def create_tenant(
        self,
        slug: str,
        name: str,
        tenant_type: str,
        owner_id: Optional[str] = None,
        **kwargs
    ) -> Tenant:
        """Create a new tenant with default configuration"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            # Get default config for tenant type
            default_config = DEFAULT_TENANT_CONFIGS.get(tenant_type, {})
            
            # Create tenant
            tenant = Tenant(
                slug=slug,
                name=name,
                tenant_type=tenant_type,
                owner_id=owner_id,
                max_users=default_config.get("max_users"),
                max_storage_gb=default_config.get("max_storage_gb"),
                max_documents=default_config.get("max_documents"),
                allowed_file_types=default_config.get("allowed_file_types", []),
                features=default_config.get("features", {}),
                **kwargs
            )
            
            session.add(tenant)
            session.commit()
            session.refresh(tenant)
            
            # Create default tenant settings
            self._create_default_tenant_settings(session, tenant.id, tenant_type)
            
            # Log tenant creation
            self._log_audit_event(
                session,
                tenant.id,
                owner_id,
                "tenant_created",
                "tenant",
                tenant.id,
                {"tenant_name": name, "tenant_type": tenant_type}
            )
            
            return tenant
        finally:
            session.close()
    
    def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            statement = select(Tenant).where(Tenant.slug == slug, Tenant.is_active == True)
            return session.exec(statement).first()
        finally:
            session.close()
    
    def get_tenant_by_id(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            statement = select(Tenant).where(Tenant.id == tenant_id, Tenant.is_active == True)
            return session.exec(statement).first()
        finally:
            session.close()
    
    def get_all_tenants(self) -> List[Tenant]:
        """Get all active tenants"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            statement = select(Tenant).where(Tenant.is_active == True)
            return list(session.exec(statement).all())
        finally:
            session.close()
    
    def add_user_to_tenant(
        self,
        tenant_id: str,
        user_id: str,
        tenant_role: str = "user",
        invited_by: Optional[str] = None,
        permissions: Optional[List[str]] = None
    ) -> TenantUser:
        """Add user to tenant with specific role"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            # Check if user is already in tenant
            existing = session.exec(
                select(TenantUser).where(
                    TenantUser.tenant_id == tenant_id,
                    TenantUser.user_id == user_id
                )
            ).first()
            
            if existing:
                raise ValueError("User is already a member of this tenant")
            
            # Get tenant to check limits
            tenant = session.get(Tenant, tenant_id)
            if not tenant:
                raise ValueError("Tenant not found")
            
            # Check user limits
            if tenant.max_users:
                current_users = session.exec(
                    select(TenantUser).where(TenantUser.tenant_id == tenant_id)
                ).all()
                if len(current_users) >= tenant.max_users:
                    raise ValueError("Tenant user limit reached")
            
            # Create tenant user relationship
            tenant_user = TenantUser(
                tenant_id=tenant_id,
                user_id=user_id,
                tenant_role=tenant_role,
                permissions=permissions or [],
                invited_by=invited_by
            )
            
            session.add(tenant_user)
            
            # Update tenant user count
            tenant.current_users += 1
            session.add(tenant)
            
            session.commit()
            session.refresh(tenant_user)
            
            # Log user addition
            self._log_audit_event(
                session,
                tenant_id,
                invited_by,
                "user_added",
                "user",
                user_id,
                {"tenant_role": tenant_role, "permissions": permissions}
            )
            
            return tenant_user
        finally:
            session.close()
    
    def remove_user_from_tenant(self, tenant_id: str, user_id: str, removed_by: Optional[str] = None):
        """Remove user from tenant"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            tenant_user = session.exec(
                select(TenantUser).where(
                    TenantUser.tenant_id == tenant_id,
                    TenantUser.user_id == user_id
                )
            ).first()
            
            if not tenant_user:
                raise ValueError("User is not a member of this tenant")
            
            session.delete(tenant_user)
            
            # Update tenant user count
            tenant = session.get(Tenant, tenant_id)
            if tenant:
                tenant.current_users = max(0, tenant.current_users - 1)
                session.add(tenant)
            
            session.commit()
            
            # Log user removal
            self._log_audit_event(
                session,
                tenant_id,
                removed_by,
                "user_removed",
                "user",
                user_id,
                {"tenant_role": tenant_user.tenant_role}
            )
        finally:
            session.close()
    
    def get_tenant_users(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get all users for a tenant with their details"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            statement = select(TenantUser, User).join(User, TenantUser.user_id == User.id).where(TenantUser.tenant_id == tenant_id)
            results = session.exec(statement).all()
            
            users = []
            for tenant_user, user in results:
                users.append({
                    "id": user.id,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "tenant_role": tenant_user.tenant_role,
                    "status": tenant_user.status,
                    "permissions": tenant_user.permissions,
                    "joined_at": tenant_user.joined_at,
                    "last_active": tenant_user.last_active,
                    "storage_used_mb": tenant_user.storage_used_mb,
                    "storage_limit_mb": tenant_user.storage_limit_mb,
                    "document_count": tenant_user.document_count,
                    "document_limit": tenant_user.document_limit
                })
            
            return users
        finally:
            session.close()
    
    def get_user_tenants(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all tenants for a user"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            statement = select(TenantUser, Tenant).join(Tenant).where(TenantUser.user_id == user_id)
            results = session.exec(statement).all()
            
            tenants = []
            for tenant_user, tenant in results:
                tenants.append({
                    "id": tenant.id,
                    "slug": tenant.slug,
                    "name": tenant.name,
                    "tenant_type": tenant.tenant_type,
                    "tenant_role": tenant_user.tenant_role,
                    "status": tenant_user.status,
                    "permissions": tenant_user.permissions,
                    "joined_at": tenant_user.joined_at,
                    "primary_color": tenant.primary_color,
                    "secondary_color": tenant.secondary_color,
                    "logo_url": tenant.logo_url
                })
            
            return tenants
        finally:
            session.close()
    
    def update_tenant_settings(
        self,
        tenant_id: str,
        settings_update: Dict[str, Any],
        updated_by: Optional[str] = None
    ):
        """Update tenant settings"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            tenant_settings = session.exec(
                select(TenantSettings).where(TenantSettings.tenant_id == tenant_id)
            ).first()
            
            if not tenant_settings:
                raise ValueError("Tenant settings not found")
            
            # Update settings
            for key, value in settings_update.items():
                if hasattr(tenant_settings, key):
                    setattr(tenant_settings, key, value)
            
            tenant_settings.updated_at = datetime.utcnow()
            session.add(tenant_settings)
            session.commit()
            
            # Log settings update
            self._log_audit_event(
                session,
                tenant_id,
                updated_by,
                "settings_updated",
                "settings",
                tenant_settings.id,
                {"updated_fields": list(settings_update.keys())}
            )
        finally:
            session.close()
    
    def get_tenant_analytics(self, tenant_id: str, days: int = 30) -> List[TenantAnalytics]:
        """Get tenant analytics for specified number of days"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            start_date = datetime.utcnow().date() - timedelta(days=days)
            statement = select(TenantAnalytics).where(
                TenantAnalytics.tenant_id == tenant_id,
                TenantAnalytics.date >= start_date
            ).order_by(TenantAnalytics.date.desc())
            
            return list(session.exec(statement).all())
        finally:
            session.close()
    
    def record_tenant_analytics(self, tenant_id: str, analytics_data: Dict[str, Any]):
        """Record daily analytics for tenant"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            today = datetime.utcnow().date()
            
            # Check if analytics already exist for today
            existing = session.exec(
                select(TenantAnalytics).where(
                    TenantAnalytics.tenant_id == tenant_id,
                    TenantAnalytics.date == today
                )
            ).first()
            
            if existing:
                # Update existing record
                for key, value in analytics_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                session.add(existing)
            else:
                # Create new record
                analytics = TenantAnalytics(
                    tenant_id=tenant_id,
                    date=today,
                    **analytics_data
                )
                session.add(analytics)
            
            session.commit()
        finally:
            session.close()
    
    def initialize_default_tenants(self):
        """Initialize the default tenants (Pepsi, KFC, Airtel, SapienPlus, Public)"""
        session_gen = get_session()
        session = next(session_gen)
        try:
            for tenant_data in INITIAL_TENANTS:
                # Check if tenant already exists
                existing = session.exec(
                    select(Tenant).where(Tenant.slug == tenant_data["slug"])
                ).first()
                
                if not existing:
                    # Get default config for tenant type
                    default_config = DEFAULT_TENANT_CONFIGS.get(tenant_data["tenant_type"], {})
                    
                    # Create tenant
                    tenant = Tenant(
                        slug=tenant_data["slug"],
                        name=tenant_data["name"],
                        tenant_type=tenant_data["tenant_type"],
                        primary_color=tenant_data.get("primary_color"),
                        secondary_color=tenant_data.get("secondary_color"),
                        max_users=default_config.get("max_users"),
                        max_storage_gb=default_config.get("max_storage_gb"),
                        max_documents=default_config.get("max_documents"),
                        allowed_file_types=default_config.get("allowed_file_types", []),
                        features=default_config.get("features", {})
                    )
                    
                    session.add(tenant)
                    session.commit()
                    session.refresh(tenant)
                    
                    # Create default tenant settings
                    self._create_default_tenant_settings(session, tenant.id, tenant_data["tenant_type"])
                    
                    print(f"Created tenant: {tenant_data['name']} ({tenant_data['slug']})")
                else:
                    print(f"Tenant already exists: {tenant_data['name']} ({tenant_data['slug']})")
        finally:
            session.close()
    
    def _create_default_tenant_settings(self, session: Session, tenant_id: str, tenant_type: str):
        """Create default settings for a tenant"""
        default_config = DEFAULT_TENANT_CONFIGS.get(tenant_type, {})
        features = default_config.get("features", {})
        
        settings = TenantSettings(
            tenant_id=tenant_id,
            enable_advanced_analytics=features.get("advanced_analytics", False),
            enable_custom_prompts=features.get("custom_prompts", False),
            enable_api_access=features.get("api_access", False),
            enable_white_labeling=features.get("white_labeling", False),
            enable_bulk_operations=features.get("bulk_operations", False),
            allow_user_uploads=(tenant_type == "public"),
            require_admin_approval=(tenant_type != "public")
        )
        
        session.add(settings)
        session.commit()
    
    def _log_audit_event(
        self,
        session: Session,
        tenant_id: str,
        user_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str],
        details: Dict[str, Any]
    ):
        """Log an audit event"""
        audit_log = TenantAuditLog(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details
        )
        
        session.add(audit_log)
        session.commit()


# Global tenant service instance
tenant_service = TenantService()
