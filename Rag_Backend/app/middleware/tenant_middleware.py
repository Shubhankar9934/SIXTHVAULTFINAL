from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
from app.auth.jwt_handler import extract_user_context_from_token
import logging

logger = logging.getLogger(__name__)

class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract and validate tenant context from JWT tokens
    and add tenant information to request state for use in route handlers
    """
    
    # Routes that don't require tenant context
    EXCLUDED_PATHS = {
        "/auth/register",
        "/auth/login", 
        "/auth/verify",
        "/auth/resend-verification",
        "/auth/forgot-password",
        "/auth/verify-reset-code",
        "/auth/validate-reset-token",
        "/auth/reset-password",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/",
        "/favicon.ico"
    }
    
    # Routes that require admin privileges within tenant
    ADMIN_ONLY_PATHS = {
        "/admin",
        "/admin/",
        "/documents/manage",
        "/users/create",
        "/users/delete",
        "/users/update"
    }
    
    # Routes that require super admin privileges (global admin)
    SUPER_ADMIN_ONLY_PATHS = {
        "/tenants",
        "/tenants/"
    }

    async def dispatch(self, request: Request, call_next):
        print(f"🔍 MIDDLEWARE DEBUG: Processing request {request.method} {request.url.path}")
        
        # Skip middleware for excluded paths
        if any(request.url.path.startswith(path) for path in self.EXCLUDED_PATHS):
            print(f"🔍 MIDDLEWARE DEBUG: Skipping excluded path: {request.url.path}")
            return await call_next(request)
        
        # Skip OPTIONS requests
        if request.method == "OPTIONS":
            print(f"🔍 MIDDLEWARE DEBUG: Skipping OPTIONS request")
            return await call_next(request)
        
        try:
            # Extract token from Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                # For public endpoints, continue without tenant context
                request.state.tenant_context = None
                return await call_next(request)
            
            token = auth_header.split(" ")[1]
            user_context = extract_user_context_from_token(token)
            
            if not user_context:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or expired token"}
                )
            
            # Create tenant context
            tenant_context = {
                "user_id": user_context["user_id"],
                "email": user_context["email"],
                "tenant_id": user_context["tenant_id"],
                "role": user_context["role"],
                "is_admin": user_context["is_admin"],
                "primary_tenant_id": user_context.get("primary_tenant_id")
            }
            
            # Validate tenant context for protected routes
            if not tenant_context["tenant_id"]:
                logger.warning(f"User {user_context['user_id']} has no tenant context")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "No tenant access. Please contact your administrator."}
                )
            
            # Check super admin-only routes (global admin required)
            is_super_admin_path = any(request.url.path.startswith(path) for path in self.SUPER_ADMIN_ONLY_PATHS)
            if is_super_admin_path:
                print(f"🔍 SUPER ADMIN PATH CHECK: Path '{request.url.path}' requires super admin access")
                print(f"🔍 SUPER ADMIN PATH CHECK: User is_admin: {tenant_context['is_admin']}")
                print(f"🔍 SUPER ADMIN PATH CHECK: User role: {tenant_context['role']}")
                print(f"🔍 SUPER ADMIN PATH CHECK: User email: {tenant_context['email']}")
                
                # For now, allow tenant admins to access tenant routes
                # TODO: Implement proper super admin logic when needed
                if not tenant_context["is_admin"]:
                    print(f"❌ SUPER ADMIN ACCESS DENIED: User {tenant_context['email']} is not admin")
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={"detail": "Super admin privileges required for this operation"}
                    )
                else:
                    print(f"✅ SUPER ADMIN ACCESS GRANTED: User {tenant_context['email']} has admin access (treating as super admin)")
            
            # Check admin-only routes (tenant admin required)
            is_admin_path = any(request.url.path.startswith(path) for path in self.ADMIN_ONLY_PATHS)
            if is_admin_path:
                print(f"🔍 ADMIN PATH CHECK: Path '{request.url.path}' requires admin access")
                print(f"🔍 ADMIN PATH CHECK: User is_admin: {tenant_context['is_admin']}")
                print(f"🔍 ADMIN PATH CHECK: User role: {tenant_context['role']}")
                print(f"🔍 ADMIN PATH CHECK: User email: {tenant_context['email']}")
                
                if not tenant_context["is_admin"]:
                    print(f"❌ ADMIN ACCESS DENIED: User {tenant_context['email']} is not admin")
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={"detail": "Admin privileges required for this operation"}
                    )
                else:
                    print(f"✅ ADMIN ACCESS GRANTED: User {tenant_context['email']} has admin access")
            
            # Add tenant context to request state
            request.state.tenant_context = tenant_context
            
            # Log tenant context for debugging (remove in production)
            print(f"🔍 MIDDLEWARE DEBUG: Tenant context set for user {tenant_context['user_id']}: tenant={tenant_context['tenant_id']}, role={tenant_context['role']}")
            print(f"🔍 MIDDLEWARE DEBUG: Request path: {request.url.path}")
            logger.debug(f"Tenant context set for user {tenant_context['user_id']}: tenant={tenant_context['tenant_id']}, role={tenant_context['role']}")
            
            return await call_next(request)
            
        except Exception as e:
            logger.error(f"Tenant middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )

def get_tenant_context(request: Request) -> Optional[dict]:
    """
    Helper function to get tenant context from request state
    """
    return getattr(request.state, 'tenant_context', None)

def require_tenant_context(request: Request) -> dict:
    """
    Helper function to get tenant context and raise error if not present
    """
    tenant_context = get_tenant_context(request)
    if not tenant_context:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return tenant_context

def require_admin_access(request: Request) -> dict:
    """
    Helper function to ensure user has admin access within their tenant
    """
    tenant_context = require_tenant_context(request)
    if not tenant_context.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return tenant_context

def validate_tenant_access(request: Request, resource_tenant_id: str) -> bool:
    """
    Validate that the current user has access to resources in the specified tenant
    """
    tenant_context = require_tenant_context(request)
    user_tenant_id = tenant_context.get("tenant_id")
    
    # Users can only access resources in their own tenant
    if user_tenant_id != resource_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Resource belongs to different tenant"
        )
    
    return True

def get_current_tenant_id(request: Request) -> str:
    """
    Get the current user's tenant ID
    """
    tenant_context = require_tenant_context(request)
    tenant_id = tenant_context.get("tenant_id")
    
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant context available"
        )
    
    return tenant_id
