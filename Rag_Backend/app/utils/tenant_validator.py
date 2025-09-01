"""
Tenant Validation Utility
========================

This module provides utilities to validate and auto-create missing tenants
to prevent foreign key constraint violations during document processing.
"""

from sqlmodel import Session, select
from typing import Optional
from app.database import get_session, User
from app.tenant_models import Tenant
from app.services.tenant_service import tenant_service
import logging

logger = logging.getLogger(__name__)

class TenantValidationError(Exception):
    """Raised when tenant validation fails"""
    pass

def validate_user_tenant(user_id: str, tenant_id: str) -> bool:
    """
    Validate that a user's tenant exists in the database.
    
    Args:
        user_id: The user's ID
        tenant_id: The tenant ID to validate
        
    Returns:
        bool: True if tenant exists, False otherwise
        
    Raises:
        TenantValidationError: If validation fails due to database errors
    """
    try:
        session_gen = get_session()
        session = next(session_gen)
        try:
            # Check if tenant exists
            tenant = session.exec(
                select(Tenant).where(
                    Tenant.id == tenant_id,
                    Tenant.is_active == True
                )
            ).first()
            
            if tenant:
                logger.info(f"✅ Tenant validation successful: {tenant_id} exists for user {user_id}")
                return True
            else:
                logger.warning(f"❌ Tenant validation failed: {tenant_id} not found for user {user_id}")
                return False
                
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"💥 Tenant validation error for user {user_id}, tenant {tenant_id}: {e}")
        raise TenantValidationError(f"Failed to validate tenant: {e}")

def auto_create_missing_tenant(user_id: str, tenant_id: str) -> bool:
    """
    Auto-create a missing tenant for a user to prevent foreign key violations.
    
    This is a recovery mechanism for cases where a user has a tenant_id
    but the tenant doesn't exist in the database.
    
    Args:
        user_id: The user's ID
        tenant_id: The tenant ID to create
        
    Returns:
        bool: True if tenant was created successfully, False otherwise
    """
    try:
        session_gen = get_session()
        session = next(session_gen)
        try:
            # Get user information
            user = session.exec(select(User).where(User.id == user_id)).first()
            if not user:
                logger.error(f"❌ Cannot create tenant: User {user_id} not found")
                return False
            
            # Check if tenant already exists (double-check)
            existing_tenant = session.exec(
                select(Tenant).where(Tenant.id == tenant_id)
            ).first()
            
            if existing_tenant:
                logger.info(f"✅ Tenant {tenant_id} already exists, no need to create")
                return True
            
            # Create a recovery tenant with the specific ID
            logger.info(f"🔧 Auto-creating missing tenant {tenant_id} for user {user.email}")
            
            # Create tenant with the exact ID that's expected
            recovery_tenant = Tenant(
                id=tenant_id,  # Use the specific tenant ID
                slug=f"recovery-{tenant_id[:8]}",  # Create a unique slug
                name=f"Recovery Tenant for {user.email}",
                tenant_type="enterprise",  # Default to enterprise type
                owner_id=user_id,
                is_active=True,
                # Set reasonable defaults
                max_users=100,
                max_storage_gb=50,
                max_documents=1000,
                allowed_file_types=["pdf", "docx", "txt", "rtf"],
                features={
                    "advanced_analytics": True,
                    "custom_prompts": True,
                    "api_access": True,
                    "white_labeling": False,
                    "bulk_operations": True
                },
                primary_color="#007bff",
                secondary_color="#6c757d"
            )
            
            session.add(recovery_tenant)
            session.commit()
            session.refresh(recovery_tenant)
            
            # Create default tenant settings
            tenant_service._create_default_tenant_settings(session, tenant_id, "enterprise")
            
            logger.info(f"✅ Successfully auto-created recovery tenant {tenant_id} for user {user.email}")
            return True
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"💥 Failed to auto-create tenant {tenant_id} for user {user_id}: {e}")
        return False

def ensure_user_tenant_exists(user_id: str, tenant_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Ensure that a user's tenant exists, creating it if necessary.
    
    Args:
        user_id: The user's ID
        tenant_id: Optional specific tenant ID to validate
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        session_gen = get_session()
        session = next(session_gen)
        try:
            # Get user information
            user = session.exec(select(User).where(User.id == user_id)).first()
            if not user:
                return False, f"User {user_id} not found"
            
            # Use provided tenant_id or user's tenant_id
            target_tenant_id = tenant_id or user.tenant_id
            
            if not target_tenant_id:
                return False, f"No tenant ID available for user {user.email}"
            
            # Validate tenant exists
            if validate_user_tenant(user_id, target_tenant_id):
                return True, f"Tenant {target_tenant_id} exists and is valid"
            
            # Try to auto-create missing tenant
            logger.warning(f"🔧 Attempting to auto-create missing tenant {target_tenant_id} for user {user.email}")
            
            if auto_create_missing_tenant(user_id, target_tenant_id):
                return True, f"Successfully created missing tenant {target_tenant_id}"
            else:
                return False, f"Failed to create missing tenant {target_tenant_id}"
                
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"💥 Error ensuring tenant exists for user {user_id}: {e}")
        return False, f"Tenant validation error: {e}"

def validate_tenant_before_processing(user_id: str, tenant_id: Optional[str] = None) -> bool:
    """
    Validate tenant before starting document processing.
    
    This function should be called before any document processing to ensure
    the tenant exists and prevent foreign key constraint violations.
    
    Args:
        user_id: The user's ID
        tenant_id: Optional specific tenant ID to validate
        
    Returns:
        bool: True if tenant is valid and processing can proceed
        
    Raises:
        TenantValidationError: If tenant validation fails and cannot be recovered
    """
    success, message = ensure_user_tenant_exists(user_id, tenant_id)
    
    if success:
        logger.info(f"✅ Tenant validation passed for user {user_id}: {message}")
        return True
    else:
        logger.error(f"❌ Tenant validation failed for user {user_id}: {message}")
        raise TenantValidationError(f"Tenant validation failed: {message}")
