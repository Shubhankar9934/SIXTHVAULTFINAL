# app/routes/tenants.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from app.database import get_session, User
from app.tenant_models import Tenant, TenantUser, TenantSettings, TenantAnalytics
from app.services.tenant_service import tenant_service
from app.deps import get_current_user

router = APIRouter(prefix="/tenants", tags=["tenants"])


# Pydantic models for request/response
class TenantCreate(BaseModel):
    slug: str
    name: str
    tenant_type: str
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    logo_url: Optional[str] = None
    custom_domain: Optional[str] = None


class TenantUpdate(BaseModel):
    name: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    logo_url: Optional[str] = None
    custom_domain: Optional[str] = None
    is_active: Optional[bool] = None


class TenantUserAdd(BaseModel):
    user_id: str
    tenant_role: str = "user"
    permissions: Optional[List[str]] = None


class TenantSettingsUpdate(BaseModel):
    allow_user_uploads: Optional[bool] = None
    require_admin_approval: Optional[bool] = None
    auto_process_documents: Optional[bool] = None
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None
    ai_temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_advanced_analytics: Optional[bool] = None
    enable_custom_prompts: Optional[bool] = None
    enable_api_access: Optional[bool] = None
    enable_white_labeling: Optional[bool] = None
    enable_bulk_operations: Optional[bool] = None
    enforce_2fa: Optional[bool] = None
    session_timeout_minutes: Optional[int] = None
    email_notifications: Optional[bool] = None


def check_super_admin(current_user: User = Depends(get_current_user)):
    """Dependency to check if user is super admin - temporarily allowing tenant admins"""
    # TODO: Implement proper super admin logic when needed
    # For now, allow tenant admins to access these routes
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def check_tenant_admin(tenant_id: str, current_user: User = Depends(get_current_user)):
    """Dependency to check if user is admin of specific tenant"""
    # TODO: Implement proper super admin logic when needed
    if current_user.is_admin:
        return current_user
    
    # Check if user is admin of this tenant
    session_gen = get_session()
    session = next(session_gen)
    try:
        tenant_user = session.exec(
            select(TenantUser).where(
                TenantUser.tenant_id == tenant_id,
                TenantUser.user_id == current_user.id,
                TenantUser.tenant_role.in_(["admin", "manager"])
            )
        ).first()
        
        if not tenant_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant admin access required"
            )
    finally:
        session.close()
    
    return current_user


@router.get("/", response_model=List[Dict[str, Any]])
async def get_all_tenants(current_user: User = Depends(check_super_admin)):
    """Get all tenants (super admin only)"""
    try:
        tenants = tenant_service.get_all_tenants()
        return [
            {
                "id": tenant.id,
                "slug": tenant.slug,
                "name": tenant.name,
                "tenant_type": tenant.tenant_type,
                "is_active": tenant.is_active,
                "current_users": tenant.current_users,
                "current_storage_mb": tenant.current_storage_mb,
                "current_documents": tenant.current_documents,
                "max_users": tenant.max_users,
                "max_storage_gb": tenant.max_storage_gb,
                "max_documents": tenant.max_documents,
                "features": tenant.features,
                "primary_color": tenant.primary_color,
                "secondary_color": tenant.secondary_color,
                "logo_url": tenant.logo_url,
                "created_at": tenant.created_at,
                "updated_at": tenant.updated_at
            }
            for tenant in tenants
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch tenants: {str(e)}"
        )


@router.post("/", response_model=Dict[str, Any])
async def create_tenant(
    tenant_data: TenantCreate,
    current_user: User = Depends(check_super_admin)
):
    """Create a new tenant (super admin only)"""
    try:
        # Check if tenant slug already exists
        existing = tenant_service.get_tenant_by_slug(tenant_data.slug)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant with this slug already exists"
            )
        
        tenant = tenant_service.create_tenant(
            slug=tenant_data.slug,
            name=tenant_data.name,
            tenant_type=tenant_data.tenant_type,
            owner_id=current_user.id,
            primary_color=tenant_data.primary_color,
            secondary_color=tenant_data.secondary_color,
            logo_url=tenant_data.logo_url,
            custom_domain=tenant_data.custom_domain
        )
        
        return {
            "id": tenant.id,
            "slug": tenant.slug,
            "name": tenant.name,
            "tenant_type": tenant.tenant_type,
            "message": "Tenant created successfully"
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tenant: {str(e)}"
        )


@router.get("/{tenant_id}", response_model=Dict[str, Any])
async def get_tenant(
    tenant_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get tenant details"""
    try:
        tenant = tenant_service.get_tenant_by_id(tenant_id)
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        # Check access permissions - allow tenant admins to access their own tenant
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Verify user belongs to this tenant (for tenant admins)
        if current_user.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this tenant"
            )
        
        return {
            "id": tenant.id,
            "slug": tenant.slug,
            "name": tenant.name,
            "tenant_type": tenant.tenant_type,
            "is_active": tenant.is_active,
            "current_users": tenant.current_users,
            "current_storage_mb": tenant.current_storage_mb,
            "current_documents": tenant.current_documents,
            "max_users": tenant.max_users,
            "max_storage_gb": tenant.max_storage_gb,
            "max_documents": tenant.max_documents,
            "allowed_file_types": tenant.allowed_file_types,
            "features": tenant.features,
            "settings": tenant.settings,
            "primary_color": tenant.primary_color,
            "secondary_color": tenant.secondary_color,
            "logo_url": tenant.logo_url,
            "custom_domain": tenant.custom_domain,
            "created_at": tenant.created_at,
            "updated_at": tenant.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch tenant: {str(e)}"
        )


@router.put("/{tenant_id}", response_model=Dict[str, Any])
async def update_tenant(
    tenant_id: str,
    tenant_update: TenantUpdate,
    current_user: User = Depends(check_super_admin)
):
    """Update tenant (super admin only)"""
    try:
        session_gen = get_session()
        session = next(session_gen)
        try:
            tenant = session.get(Tenant, tenant_id)
            if not tenant:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Tenant not found"
                )
            
            # Update fields
            update_data = tenant_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(tenant, field, value)
            
            from datetime import datetime
            tenant.updated_at = datetime.utcnow()
            session.add(tenant)
            session.commit()
            session.refresh(tenant)
            
            return {
                "id": tenant.id,
                "message": "Tenant updated successfully"
            }
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update tenant: {str(e)}"
        )


@router.get("/{tenant_id}/users", response_model=List[Dict[str, Any]])
async def get_tenant_users(
    tenant_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all users for a tenant"""
    try:
        # Check access permissions - allow tenant admins to access their own tenant
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Verify user belongs to this tenant (for tenant admins)
        if current_user.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this tenant"
            )
        
        users = tenant_service.get_tenant_users(tenant_id)
        return users
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch tenant users: {str(e)}"
        )


@router.post("/{tenant_id}/users", response_model=Dict[str, Any])
async def add_user_to_tenant(
    tenant_id: str,
    user_data: TenantUserAdd,
    current_user: User = Depends(get_current_user)
):
    """Add user to tenant"""
    try:
        # Check access permissions - allow tenant admins to access their own tenant
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Verify user belongs to this tenant (for tenant admins)
        if current_user.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this tenant"
            )
        
        tenant_user = tenant_service.add_user_to_tenant(
            tenant_id=tenant_id,
            user_id=user_data.user_id,
            tenant_role=user_data.tenant_role,
            invited_by=current_user.id,
            permissions=user_data.permissions
        )
        
        return {
            "id": tenant_user.id,
            "message": "User added to tenant successfully"
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add user to tenant: {str(e)}"
        )


@router.delete("/{tenant_id}/users/{user_id}", response_model=Dict[str, Any])
async def remove_user_from_tenant(
    tenant_id: str,
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """Remove user from tenant"""
    try:
        # Check access permissions
        if not current_user.is_super_admin:
            session_gen = get_session()
            session = next(session_gen)
            try:
                tenant_user = session.exec(
                    select(TenantUser).where(
                        TenantUser.tenant_id == tenant_id,
                        TenantUser.user_id == current_user.id,
                        TenantUser.tenant_role.in_(["admin", "manager"])
                    )
                ).first()
                
                if not tenant_user:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Tenant admin access required"
                    )
            finally:
                session.close()
        
        tenant_service.remove_user_from_tenant(
            tenant_id=tenant_id,
            user_id=user_id,
            removed_by=current_user.id
        )
        
        return {"message": "User removed from tenant successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove user from tenant: {str(e)}"
        )


@router.get("/{tenant_id}/settings", response_model=Dict[str, Any])
async def get_tenant_settings(
    tenant_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get tenant settings"""
    try:
        # Check access permissions
        if not current_user.is_super_admin:
            session_gen = get_session()
            session = next(session_gen)
            try:
                tenant_user = session.exec(
                    select(TenantUser).where(
                        TenantUser.tenant_id == tenant_id,
                        TenantUser.user_id == current_user.id
                    )
                ).first()
                
                if not tenant_user:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to this tenant"
                    )
            finally:
                session.close()
        
        session_gen = get_session()
        session = next(session_gen)
        try:
            settings = session.exec(
                select(TenantSettings).where(TenantSettings.tenant_id == tenant_id)
            ).first()
            
            if not settings:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Tenant settings not found"
                )
        finally:
            session.close()
            
            return {
                "id": settings.id,
                "tenant_id": settings.tenant_id,
                "allow_user_uploads": settings.allow_user_uploads,
                "require_admin_approval": settings.require_admin_approval,
                "auto_process_documents": settings.auto_process_documents,
                "ai_provider": settings.ai_provider,
                "ai_model": settings.ai_model,
                "ai_temperature": settings.ai_temperature,
                "max_tokens": settings.max_tokens,
                "enable_advanced_analytics": settings.enable_advanced_analytics,
                "enable_custom_prompts": settings.enable_custom_prompts,
                "enable_api_access": settings.enable_api_access,
                "enable_white_labeling": settings.enable_white_labeling,
                "enable_bulk_operations": settings.enable_bulk_operations,
                "enforce_2fa": settings.enforce_2fa,
                "session_timeout_minutes": settings.session_timeout_minutes,
                "email_notifications": settings.email_notifications,
                "created_at": settings.created_at,
                "updated_at": settings.updated_at
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch tenant settings: {str(e)}"
        )


@router.put("/{tenant_id}/settings", response_model=Dict[str, Any])
async def update_tenant_settings(
    tenant_id: str,
    settings_update: TenantSettingsUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update tenant settings"""
    try:
        # Check access permissions
        if not current_user.is_super_admin:
            session_gen = get_session()
            session = next(session_gen)
            try:
                tenant_user = session.exec(
                    select(TenantUser).where(
                        TenantUser.tenant_id == tenant_id,
                        TenantUser.user_id == current_user.id,
                        TenantUser.tenant_role.in_(["admin", "manager"])
                    )
                ).first()
                
                if not tenant_user:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Tenant admin access required"
                    )
            finally:
                session.close()
        
        update_data = settings_update.dict(exclude_unset=True)
        tenant_service.update_tenant_settings(
            tenant_id=tenant_id,
            settings_update=update_data,
            updated_by=current_user.id
        )
        
        return {"message": "Tenant settings updated successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update tenant settings: {str(e)}"
        )


@router.get("/{tenant_id}/analytics", response_model=List[Dict[str, Any]])
async def get_tenant_analytics(
    tenant_id: str,
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Get tenant analytics"""
    try:
        # Check access permissions
        if not current_user.is_super_admin:
            session_gen = get_session()
            session = next(session_gen)
            try:
                tenant_user = session.exec(
                    select(TenantUser).where(
                        TenantUser.tenant_id == tenant_id,
                        TenantUser.user_id == current_user.id,
                        TenantUser.tenant_role.in_(["admin", "manager"])
                    )
                ).first()
                
                if not tenant_user:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Tenant admin access required"
                    )
            finally:
                session.close()
        
        analytics = tenant_service.get_tenant_analytics(tenant_id, days)
        return [
            {
                "date": str(record.date),
                "active_users": record.active_users,
                "new_users": record.new_users,
                "total_users": record.total_users,
                "documents_uploaded": record.documents_uploaded,
                "documents_processed": record.documents_processed,
                "total_documents": record.total_documents,
                "storage_used_mb": record.storage_used_mb,
                "ai_queries": record.ai_queries,
                "ai_tokens_used": record.ai_tokens_used,
                "curations_generated": record.curations_generated,
                "summaries_generated": record.summaries_generated,
                "avg_response_time_ms": record.avg_response_time_ms,
                "error_rate": record.error_rate,
                "uptime_percentage": record.uptime_percentage
            }
            for record in analytics
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch tenant analytics: {str(e)}"
        )


@router.post("/initialize", response_model=Dict[str, Any])
async def initialize_default_tenants(current_user: User = Depends(check_super_admin)):
    """Initialize default tenants (super admin only)"""
    try:
        tenant_service.initialize_default_tenants()
        return {"message": "Default tenants initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize tenants: {str(e)}"
        )


@router.get("/user/{user_id}/tenants", response_model=List[Dict[str, Any]])
async def get_user_tenants(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all tenants for a user"""
    try:
        # Users can only see their own tenants unless they're admin
        if not current_user.is_admin and current_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        tenants = tenant_service.get_user_tenants(user_id)
        return tenants
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user tenants: {str(e)}"
        )
