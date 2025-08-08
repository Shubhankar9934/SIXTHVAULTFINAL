"""
JWT Authentication and dependency injection
"""
from fastapi import Depends, HTTPException, status, Header
from sqlmodel import Session, select
from typing import Optional
from app.database import get_session
from app.database import User
from app.auth.jwt_handler import verify_token
from app.auth.token_service import TokenService

async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_session)
) -> User:
    """Get current authenticated user from JWT token"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Extract token from "Bearer <token>"
        parts = authorization.split()
        if len(parts) != 2:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        scheme, token = parts
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify token in database first
    user_id = await TokenService.validate_token(db, token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Validate JWT signature and expiry as additional security
    payload = verify_token(token)
    if not payload or payload.get("sub") != user_id:
        # If JWT validation fails or user ID mismatch, revoke the token
        await TokenService.revoke_token(db, token)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database using validated user_id
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.exec(select(User).where(User.id == user_id)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User email not verified",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
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

# For backward compatibility and database access
get_db = get_session
