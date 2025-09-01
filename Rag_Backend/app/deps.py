"""
JWT Authentication and dependency injection with multi-tenant support
"""
from fastapi import Depends, HTTPException, status, Header, Request
from sqlmodel import Session, select
from typing import Optional
from app.database import get_session
from app.database import User
from app.auth.jwt_handler import verify_token
from app.auth.token_service import TokenService
from app.middleware.tenant_middleware import (
    get_tenant_context, require_tenant_context, require_admin_access,
    validate_tenant_access, get_current_tenant_id
)

async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    db: Session = Depends(get_session)
) -> User:
    """Get current authenticated user from JWT token - Enhanced with debugging"""
    print(f"🔍 AUTH DEBUG: Starting authentication process")
    print(f"🔍 AUTH DEBUG: Authorization header received: {authorization is not None}")
    
    if not authorization:
        print(f"🔐 AUTH: No authorization header provided")
        print(f"🔍 AUTH DEBUG: Request headers available but no Authorization header found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed - please log in again",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    print(f"🔍 AUTH DEBUG: Authorization header length: {len(authorization)}")
    print(f"🔍 AUTH DEBUG: Authorization header preview: {authorization[:50]}...")
    
    try:
        # Extract token from "Bearer <token>" with improved parsing
        if not authorization.startswith("Bearer "):
            print(f"🔐 AUTH: Invalid authorization header format")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed - please log in again",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token = authorization[7:]  # Remove "Bearer " prefix
        if not token or token.strip() == "":
            print(f"🔐 AUTH: Empty token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed - please log in again",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"🔐 AUTH: Header parsing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed - please log in again",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Validate JWT signature and expiry
        payload = verify_token(token)
        if not payload:
            print(f"🔐 AUTH: JWT validation failed - token invalid or expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed - please log in again",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("sub")
        if not user_id:
            print(f"🔐 AUTH: No user ID in JWT payload")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed - please log in again",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database - simplified approach
        user = db.exec(select(User).where(User.id == user_id)).first()
        if not user:
            print(f"🔐 AUTH: User not found in database: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed - please log in again",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is active and verified
        if not user.is_active:
            print(f"🔐 AUTH: User account deactivated: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.verified:
            print(f"🔐 AUTH: User account not verified: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is not verified",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Optional database token validation (non-blocking)
        try:
            db_user_id = await TokenService.validate_token(db, token)
            if db_user_id and db_user_id != user_id:
                print(f"🔐 AUTH: Token validation mismatch, but proceeding with JWT validation")
            elif db_user_id:
                print(f"🔐 AUTH: Database token validation successful for user {user_id}")
        except Exception as db_error:
            print(f"🔐 AUTH: Database token validation failed (non-critical): {db_error}")
            # Continue with JWT-based authentication
        
        print(f"✅ AUTH: Successfully authenticated user: {user.email}")
        return user
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"💥 AUTH: Unexpected authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed - please log in again",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user_optional(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    db: Session = Depends(get_session)
) -> Optional[User]:
    """Get current authenticated user from JWT token, return None if not authenticated"""
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>"
        parts = authorization.split()
        if len(parts) != 2:
            return None
        
        scheme, token = parts
        if scheme.lower() != "bearer":
            return None
        
        # Verify token in database first
        user_id = await TokenService.validate_token(db, token)
        if not user_id:
            return None
        
        # Validate JWT signature and expiry as additional security
        payload = verify_token(token)
        if not payload or payload.get("sub") != user_id:
            # If JWT validation fails or user ID mismatch, revoke the token
            await TokenService.revoke_token(db, token)
            return None
        
        # Get user from database using validated user_id
        user = db.exec(select(User).where(User.id == user_id)).first()
        if not user or not user.verified or not user.is_active:
            return None
        
        return user
        
    except Exception:
        return None

def get_current_user_from_id(user_id: str, db: Session) -> Optional[User]:
    """Get user by ID for pre-caching operations"""
    try:
        user = db.exec(select(User).where(User.id == user_id)).first()
        if not user or not user.verified or not user.is_active:
            return None
        return user
    except Exception as e:
        print(f"Failed to get user by ID {user_id}: {e}")
        return None

# Tenant-aware dependencies
async def get_current_user_with_tenant(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user with tenant context validation - fallback to user's tenant_id if middleware context not available"""
    print(f"🔍 TENANT DEBUG: Checking tenant context for user {current_user.email}")
    
    tenant_context = get_tenant_context(request)
    print(f"🔍 TENANT DEBUG: Tenant context from request: {tenant_context}")
    print(f"🔍 TENANT DEBUG: User tenant_id: {current_user.tenant_id}")
    
    if not tenant_context:
        print(f"⚠️ TENANT DEBUG: No tenant context found in request, using user's tenant_id as fallback")
        print(f"🔍 TENANT DEBUG: Request state: {getattr(request.state, 'tenant_context', 'NOT_SET')}")
        
        # Fallback: If middleware didn't set tenant context, use user's tenant_id
        if not current_user.tenant_id:
            print(f"❌ TENANT DEBUG: User has no tenant_id either")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No tenant access. Please contact your administrator."
            )
        
        # Create a fallback tenant context from user data
        fallback_context = {
            "user_id": current_user.id,
            "email": current_user.email,
            "tenant_id": current_user.tenant_id,
            "role": current_user.role,
            "is_admin": current_user.is_admin,
            "primary_tenant_id": current_user.primary_tenant_id
        }
        
        # Set the fallback context in request state for other dependencies
        request.state.tenant_context = fallback_context
        print(f"✅ TENANT DEBUG: Set fallback tenant context for user {current_user.email}")
        return current_user
    
    # Validate user belongs to the tenant
    if current_user.tenant_id != tenant_context["tenant_id"]:
        print(f"❌ TENANT DEBUG: User tenant mismatch - user: {current_user.tenant_id}, context: {tenant_context['tenant_id']}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not belong to the current tenant"
        )
    
    print(f"✅ TENANT DEBUG: Tenant validation successful for user {current_user.email}")
    return current_user

async def get_current_admin_user(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user and ensure they have admin privileges"""
    print(f"🔍 ADMIN AUTH: Checking admin access for user {current_user.email}")
    print(f"🔍 ADMIN AUTH: User is_admin: {current_user.is_admin}")
    print(f"🔍 ADMIN AUTH: User role: {current_user.role}")
    print(f"🔍 ADMIN AUTH: User tenant_id: {current_user.tenant_id}")
    
    # Double-check user is admin
    if not current_user.is_admin:
        print(f"❌ ADMIN AUTH: User {current_user.email} is not admin")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    # Try to get tenant context, but don't fail if it's not available
    try:
        tenant_context = require_admin_access(request)
        print(f"✅ ADMIN AUTH: Tenant context found: {tenant_context}")
        
        # Validate user belongs to the tenant
        if current_user.tenant_id != tenant_context["tenant_id"]:
            print(f"❌ ADMIN AUTH: Tenant mismatch - user: {current_user.tenant_id}, context: {tenant_context['tenant_id']}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin does not belong to the current tenant"
            )
    except HTTPException as e:
        print(f"⚠️ ADMIN AUTH: Tenant context not available, using fallback approach")
        # Fallback: If middleware didn't set tenant context, create one from user data
        if not current_user.tenant_id:
            print(f"❌ ADMIN AUTH: User has no tenant_id")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tenant access. Please contact your administrator."
            )
        
        # Create fallback tenant context
        fallback_context = {
            "user_id": current_user.id,
            "email": current_user.email,
            "tenant_id": current_user.tenant_id,
            "role": current_user.role,
            "is_admin": current_user.is_admin,
            "primary_tenant_id": current_user.primary_tenant_id
        }
        
        # Set the fallback context in request state
        request.state.tenant_context = fallback_context
        print(f"✅ ADMIN AUTH: Set fallback tenant context for admin {current_user.email}")
    
    print(f"✅ ADMIN AUTH: Admin access granted for user {current_user.email}")
    return current_user

def get_tenant_context_dependency(request: Request) -> dict:
    """Dependency to get tenant context from request"""
    return require_tenant_context(request)

def get_current_tenant_id_dependency(request: Request) -> str:
    """Dependency to get current tenant ID - with fallback support"""
    try:
        return get_current_tenant_id(request)
    except HTTPException:
        # If tenant context is not available, try to get it from request state
        tenant_context = getattr(request.state, 'tenant_context', None)
        if tenant_context and tenant_context.get('tenant_id'):
            return tenant_context['tenant_id']
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No tenant context available"
        )

# For backward compatibility and database access
get_db = get_session
